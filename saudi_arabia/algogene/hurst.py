from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from datetime import datetime, timedelta
import numpy as np


class AlgoEvent:
    # ───────────────────────── CONFIG ──────────────────────────
    def __init__(self):
        # base size for Gold leg
        self.base_lot = 3

        # timers & bookkeeping
        self.lasttradetime = datetime(1900, 1, 1)
        self.orderPairCnt = 0
        self.osOrder: dict = {}
        self.contractSize = {}
        self.minVolume = {}
        self.volumeIncrement = {}
        self.entry_time = {}  # track entry timestamp per pair

        # keep track of regime
        self.current_regime = "neutral"
        self.regime_change_time = None
        self.regime_stability_window = 20
        self.min_regime_duration = 5  # hours
        self.regime_history = []

        # indicator windows
        self.hurst_window = 30  # bars (shorter for aggressive)
        self.signal_window = 10
        self.mom_window = 10

        # BASE MR thresholds (will be adjusted dynamically)
        self.base_mr_hurst_thresh = 0.50
        self.base_z_entry_mr = 2.0
        self.base_z_exit_mr = 1.0
        self.max_hold_hours = 12

        # BASE TF thresholds (will be adjusted dynamically)
        self.base_tf_hurst_thresh = 0.50
        self.base_mom_entry_tf = 1.20
        self.base_mom_exit_tf = 0.20

        # Current dynamic thresholds (initialized to base values)
        self.mr_hurst_thresh = self.base_mr_hurst_thresh
        self.z_entry_mr = self.base_z_entry_mr
        self.z_exit_mr = self.base_z_exit_mr
        self.tf_hurst_thresh = self.base_tf_hurst_thresh
        self.mom_entry_tf = self.base_mom_entry_tf
        self.mom_exit_tf = self.base_mom_exit_tf

        # data buffers
        self.arr_gold = []
        self.arr_silver = []
        self.arr_spread = []
        self.arr_filtered_spread = []

        # trade counters
        self.total_trades = 0
        self.mr_trades = 0
        self.tf_trades = 0

        # Risk Management
        self.max_drawdown_pct = 0.15
        self.trailing_stop_pct = 0.08
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.volatility_lookback = 20
        self.base_volatility = 0.02

        # Position tracking for risk management
        self.position_entry_equity = {}
        self.peak_position_value = {}

        # Kalman Filter for spread
        self.kf_initialized = False
        self.kf_state = np.array([0.0, 0.0])  # [spread, spread_velocity]
        self.kf_covariance = np.eye(2) * 0.1
        self.kf_process_noise = 0.01
        self.kf_measurement_noise = 0.1

    # ───────────────────────── STARTUP ─────────────────────────
    def start(self, mEvt):
        self.gold, self.silver = mEvt["subscribeList"]
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)

        # fetch specs
        for sym in (self.gold, self.silver):
            spec = self.evt.getContractSpec(sym)
            self.contractSize[sym] = spec["contractSize"]
            self.minVolume[sym] = spec.get("minVolume", 1)
            self.volumeIncrement[sym] = spec.get("volumeIncrement", 1)

        self.evt.consoleLog("Starting Enhanced Dual-Regime Strategy with Dynamic Parameters")
        self.evt.consoleLog(f"Gold (Y): {self.gold}, Silver (X): {self.silver}")
        self.evt.start()
        self.evt.on_dailyDumpData(self)

    # ───────────────────────── KALMAN FILTER ───────────────────
    def _update_kalman_filter(self, raw_spread):
        """Update Kalman filter for spread smoothing"""
        if not self.kf_initialized:
            self.kf_state[0] = raw_spread
            self.kf_initialized = True
            return raw_spread

        # Prediction step
        # State transition matrix (constant velocity model)
        F = np.array([[1.0, 1.0],
                      [0.0, 1.0]])

        # Process noise covariance
        Q = np.array([[self.kf_process_noise, 0.0],
                      [0.0, self.kf_process_noise]])

        # Predict
        self.kf_state = F @ self.kf_state
        self.kf_covariance = F @ self.kf_covariance @ F.T + Q

        # Update step
        # Measurement matrix (we observe the spread directly)
        H = np.array([[1.0, 0.0]])

        # Measurement noise covariance
        R = np.array([[self.kf_measurement_noise]])

        # Innovation
        y = raw_spread - H @ self.kf_state
        S = H @ self.kf_covariance @ H.T + R
        K = self.kf_covariance @ H.T / S

        # Update
        self.kf_state = self.kf_state + K * y
        I_KH = np.eye(2) - K @ H
        self.kf_covariance = I_KH @ self.kf_covariance

        return self.kf_state[0]

    # ───────────────────────── REGIME STABILITY ───────────────────
    def _stable_regime_check(self, new_regime):
        """Check if regime change is stable enough to act upon"""
        if self.current_regime == new_regime:
            return True

        # Add to regime history
        self.regime_history.append((datetime.now(), new_regime))

        # Keep only recent history
        if len(self.regime_history) > self.regime_stability_window:
            self.regime_history.pop(0)

        # Check if we have enough history
        if len(self.regime_history) < self.min_regime_duration:
            return False

        # Check if the new regime has been consistent for minimum duration
        recent_regimes = [r[1] for r in self.regime_history[-self.min_regime_duration:]]
        regime_stability = sum(1 for r in recent_regimes if r == new_regime) / len(recent_regimes)

        # Require 80% consistency for regime change
        if regime_stability >= 0.8:
            if self.regime_change_time is None:
                self.regime_change_time = datetime.now()

            time_elapsed = datetime.now() - self.regime_change_time
            if time_elapsed >= timedelta(hours=self.min_regime_duration):
                return True
        else:
            self.regime_change_time = None

        return False

    # ───────────────────────── DYNAMIC PARAMETERS ───────────────────
    def _calculate_market_volatility(self):
        """Calculate recent market volatility for dynamic parameter adjustment"""
        if len(self.arr_spread) < self.volatility_lookback:
            return self.base_volatility

        recent_spreads = self.arr_spread[-self.volatility_lookback:]
        returns = np.diff(recent_spreads) / np.abs(recent_spreads[:-1])
        returns = returns[np.isfinite(returns)]  # Remove inf/nan

        if len(returns) == 0:
            return self.base_volatility

        volatility = np.std(returns) if len(returns) > 1 else self.base_volatility
        return max(0.001, volatility)  # Minimum volatility floor

    def _update_dynamic_parameters(self):
        """Update trading parameters based on current market conditions"""
        current_vol = self._calculate_market_volatility()
        vol_ratio = current_vol / self.base_volatility

        # Volatility regime classification
        if vol_ratio > 2.0:
            volatility_regime = "high"
        elif vol_ratio < 0.5:
            volatility_regime = "low"
        else:
            volatility_regime = "normal"

        # Adjust MR parameters
        if volatility_regime == "high":
            self.z_entry_mr = self.base_z_entry_mr * 1.5  # Wider entry bands
            self.z_exit_mr = self.base_z_exit_mr * 1.3
            self.mr_hurst_thresh = self.base_mr_hurst_thresh - 0.05  # Easier to enter MR
        elif volatility_regime == "low":
            self.z_entry_mr = self.base_z_entry_mr * 0.7  # Tighter entry bands
            self.z_exit_mr = self.base_z_exit_mr * 0.8
            self.mr_hurst_thresh = self.base_mr_hurst_thresh + 0.05  # Harder to enter MR
        else:
            self.z_entry_mr = self.base_z_entry_mr
            self.z_exit_mr = self.base_z_exit_mr
            self.mr_hurst_thresh = self.base_mr_hurst_thresh

        # Adjust TF parameters
        if volatility_regime == "high":
            self.mom_entry_tf = self.base_mom_entry_tf * 1.4  # Need stronger momentum
            self.mom_exit_tf = self.base_mom_exit_tf * 1.2
            self.tf_hurst_thresh = self.base_tf_hurst_thresh + 0.05  # Harder to enter TF
        elif volatility_regime == "low":
            self.mom_entry_tf = self.base_mom_entry_tf * 0.8  # Lower momentum threshold
            self.mom_exit_tf = self.base_mom_exit_tf * 0.9
            self.tf_hurst_thresh = self.base_tf_hurst_thresh - 0.05  # Easier to enter TF
        else:
            self.mom_entry_tf = self.base_mom_entry_tf
            self.mom_exit_tf = self.base_mom_exit_tf
            self.tf_hurst_thresh = self.base_tf_hurst_thresh

    # ───────────────────────── RISK MANAGEMENT ───────────────────
    def _calculate_volatility_adjusted_size(self):
        """Calculate position size based on current volatility"""
        current_vol = self._calculate_market_volatility()
        vol_adjust = min(2.0, max(0.5, self.base_volatility / current_vol))
        return self.base_lot * vol_adjust

    def _check_drawdown_limits(self):
        """Check if we're within acceptable drawdown limits"""
        if self.peak_equity == 0:
            return True

        current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        return current_drawdown < self.max_drawdown_pct

    def _update_risk_metrics(self):
        """Update risk tracking metrics"""
        # This would be called from position feed or P&L updates
        # For now, we'll approximate using theoretical calculations
        total_positions = len(self.osOrder)
        if total_positions > 0:
            # Estimate current equity (simplified)
            self.current_equity = max(self.current_equity, 100000)  # Starting equity assumption
            self.peak_equity = max(self.peak_equity, self.current_equity)

    def _position_risk_check(self, orderRef):
        """Check individual position risk"""
        if orderRef not in self.position_entry_equity:
            return True

        # Check trailing stop
        entry_equity = self.position_entry_equity[orderRef]
        peak_value = self.peak_position_value.get(orderRef, entry_equity)

        # Update peak if current is higher
        current_value = self.current_equity  # Simplified
        if current_value > peak_value:
            self.peak_position_value[orderRef] = current_value
            peak_value = current_value

        # Check trailing stop
        trailing_loss = (peak_value - current_value) / peak_value
        return trailing_loss < self.trailing_stop_pct

    # ───────────────────────── DATA LOOP ───────────────────────
    def on_bulkdatafeed(self, isSync, bd, ab):
        if not isSync:
            return

        now = bd[self.gold]["timestamp"]
        # throttle to one update per hourly bar
        if now < self.lasttradetime + timedelta(hours=1):
            return
        self.lasttradetime = now

        # fetch prices
        p_g = bd[self.gold]["lastPrice"]
        p_s = bd[self.silver]["lastPrice"]

        # update buffers
        self.arr_gold.append(p_g)
        self.arr_silver.append(p_s)
        if len(self.arr_gold) > self.hurst_window:   self.arr_gold.pop(0)
        if len(self.arr_silver) > self.hurst_window:   self.arr_silver.pop(0)
        if len(self.arr_gold) < self.hurst_window:   return  # warming up

        # compute raw log-spread
        raw_spread = np.log(p_g) - np.log(p_s)

        # Apply Kalman filter for smoothing
        filtered_spread = self._update_kalman_filter(raw_spread)

        self.arr_spread.append(raw_spread)
        self.arr_filtered_spread.append(filtered_spread)

        if len(self.arr_spread) > self.hurst_window:
            self.arr_spread.pop(0)
        if len(self.arr_filtered_spread) > self.hurst_window:
            self.arr_filtered_spread.pop(0)

        # Update dynamic parameters based on market conditions
        self._update_dynamic_parameters()

        # Update risk metrics
        self._update_risk_metrics()

        # Check if we should stop trading due to drawdown
        if not self._check_drawdown_limits():
            self.evt.consoleLog(">>> DRAWDOWN LIMIT REACHED - HALTING NEW TRADES")
            # Close all positions
            for leg1, leg2 in self.matchPairTradeID().items():
                self.closeOrder(leg1)
                self.closeOrder(leg2)
            return

        # Use filtered spread for calculations
        spread_data = self.arr_filtered_spread[-self.hurst_window:]

        # indicators
        hurst = self._calc_hurst(spread_data)
        zscore = self._calc_zscore(self.arr_filtered_spread[-self.signal_window:])
        momentum = (self._calc_momentum(self.arr_filtered_spread[-self.mom_window:])
                    if len(self.arr_filtered_spread) >= self.mom_window else 0.0)

        # regime flags with dynamic thresholds
        can_mr = hurst < self.mr_hurst_thresh
        can_tf = hurst > self.tf_hurst_thresh

        # Determine potential new regime
        if can_mr:
            potential_regime = "mr"
        elif can_tf:
            potential_regime = "tf"
        else:
            potential_regime = "neutral"

        # Check regime stability before making changes
        if self._stable_regime_check(potential_regime):
            if self.current_regime != potential_regime:
                self.evt.consoleLog(f">>> REGIME CHANGE: {self.current_regime} -> {potential_regime}")
                self.current_regime = potential_regime
                self.regime_change_time = now

        # ENTRY
        if not self.matchPairTradeID():
            volatility_adjusted_size = self._calculate_volatility_adjusted_size()

            if self.current_regime == "mr" and abs(zscore) > self.z_entry_mr:
                dir_ = -1 if zscore > 0 else +1
                self._open_pair(dir_, "mr", now, volatility_adjusted_size)
            elif self.current_regime == "tf":
                if momentum > self.mom_entry_tf:
                    self._open_pair(+1, "tf", now, volatility_adjusted_size)
                elif momentum < -self.mom_entry_tf:
                    self._open_pair(-1, "tf", now, volatility_adjusted_size)

        # EXIT
        for leg1, leg2 in self.matchPairTradeID().items():
            o1 = self.osOrder.get(leg1)
            if not o1:
                continue

            orderRef = o1['orderRef']

            # Check position-specific risk limits
            if not self._position_risk_check(orderRef):
                self.evt.consoleLog(f">>> CLOSING due to trailing stop (Ref={orderRef})")
                self.closeOrder(leg1)
                self.closeOrder(leg2)
                continue

            # Mean-Reversion exit
            if self.current_regime == "mr":
                held = now - self.entry_time.get(orderRef, now)
                if abs(zscore) < self.z_exit_mr or held > timedelta(hours=self.max_hold_hours):
                    self.evt.consoleLog(f">>> CLOSING MR (z={zscore:.2f}, held={held})")
                    self.closeOrder(leg1)
                    self.closeOrder(leg2)

            # Trend-Following exit
            elif self.current_regime == "tf":
                bs1 = o1['buysell']
                gold_side = bs1 if o1['instrument'] == self.gold else -bs1
                if (gold_side == 1 and momentum < self.mom_exit_tf) or \
                        (gold_side == -1 and momentum > -self.mom_exit_tf):
                    self.evt.consoleLog(f">>> CLOSING TF (mom={momentum:.2f})")
                    self.closeOrder(leg1)
                    self.closeOrder(leg2)

        # LOG with enhanced information
        open_ct = len(self.matchPairTradeID())
        current_vol = self._calculate_market_volatility()
        self.evt.consoleLog(
            f"t={now:%Y-%m-%d %H:%M} H={hurst:.3f} Z={zscore:.2f} M={momentum:.2f} "
            f"Vol={current_vol:.4f} Regime={self.current_regime} Open={open_ct} "
            f"Z_entry={self.z_entry_mr:.2f} Mom_entry={self.mom_entry_tf:.2f} "
            f"Total={self.total_trades} MR={self.mr_trades} TF={self.tf_trades}"
        )

    # ─────────────────── ENHANCED PAIR OPENING ───────────────────
    def _open_pair(self, direction, regime, now, vol_multiplier=1.0):
        self.orderPairCnt += 1
        self.total_trades += 1
        if regime == "mr":
            self.mr_trades += 1
        else:
            self.tf_trades += 1

        # Gold leg: adjusted size with volatility multiplier
        adjusted_base_lot = self.base_lot * vol_multiplier
        inc_g = self.volumeIncrement[self.gold]
        mv_g = self.minVolume[self.gold]
        vol_g = max(mv_g, round(adjusted_base_lot / inc_g) * inc_g)

        # Silver leg: dollar-neutral (β=1)
        price_g, price_s = self.arr_gold[-1], self.arr_silver[-1]
        cs_g, cs_s = self.contractSize[self.gold], self.contractSize[self.silver]
        notional_g = vol_g * price_g * cs_g
        raw_s = notional_g / (price_s * cs_s)
        inc_s, mv_s = self.volumeIncrement[self.silver], self.minVolume[self.silver]
        vol_s = max(mv_s, round(raw_s / inc_s) * inc_s)

        # Record position entry for risk management
        self.position_entry_equity[self.orderPairCnt] = self.current_equity
        self.peak_position_value[self.orderPairCnt] = self.current_equity

        # send orders
        self.openOrder(direction, self.gold, self.orderPairCnt, vol_g)
        self.openOrder(-direction, self.silver, self.orderPairCnt, vol_s)

        # record entry time
        self.entry_time[self.orderPairCnt] = now

        self.evt.consoleLog(
            f">>> OPEN {regime.upper()} dir={direction:+d} "
            f"GoldVol={vol_g} SilverVol={vol_s} VolMult={vol_multiplier:.2f}"
        )

    # ───────────── order & helper methods ─────────────
    def matchPairTradeID(self):
        pairs = {}
        for tid, od in self.osOrder.items():
            ref = od['orderRef']
            for tid2, od2 in self.osOrder.items():
                if tid != tid2 and od2['orderRef'] == ref and tid not in pairs:
                    pairs[tid] = tid2
                    break
        return pairs

    def closeOrder(self, tradeID):
        # Clean up risk tracking when closing
        if tradeID in self.osOrder:
            orderRef = self.osOrder[tradeID]['orderRef']
            if orderRef in self.position_entry_equity:
                del self.position_entry_equity[orderRef]
            if orderRef in self.peak_position_value:
                del self.peak_position_value[orderRef]

        self.evt.sendOrder(AlgoAPIUtil.OrderObject(tradeID=tradeID, openclose='close'))

    def openOrder(self, buysell, instrument, orderRef, volume):
        self.evt.sendOrder(AlgoAPIUtil.OrderObject(
            instrument=instrument,
            orderRef=orderRef,
            volume=volume,
            openclose='open',
            buysell=buysell,
            ordertype=0  # market
        ))

    # ───────────────── required callback stubs ─────────────────
    def on_marketdatafeed(self, md, ab):
        pass

    def on_orderfeed(self, of):
        pass

    def on_dailyPLfeed(self, pl):
        # Update equity tracking for risk management
        if 'totalPL' in pl:
            self.current_equity = pl['totalPL']
            self.peak_equity = max(self.peak_equity, self.current_equity)

    def on_openPositionfeed(self, op, oo, uo):
        self.osOrder = oo

    def on_dailyDumpData(self, dump):
        # Enhanced daily dump with risk metrics
        if hasattr(self, 'current_equity') and self.peak_equity > 0:
            current_dd = (self.peak_equity - self.current_equity) / self.peak_equity
            self.evt.consoleLog(f"Daily Risk Summary: DD={current_dd:.1%}, Peak={self.peak_equity:.0f}")

    # ───────────────── indicator helpers ─────────────────
    @staticmethod
    def _calc_zscore(arr):
        if len(arr) < 2:
            return 0.0
        μ, σ = np.mean(arr), np.std(arr, ddof=1)
        return 0.0 if σ == 0 else (arr[-1] - μ) / σ

    @staticmethod
    def _calc_momentum(arr):
        if len(arr) < 3:
            return 0.0
        rets = [(arr[i] - arr[i - 1]) / abs(arr[i - 1])
                for i in range(1, len(arr)) if arr[i - 1] != 0]
        if len(rets) < 2:
            return 0.0
        μ, σ = np.mean(rets), np.std(rets, ddof=1)
        return 0.0 if σ == 0 else μ / σ

    @staticmethod
    def _calc_hurst(series):
        ts = np.array(series) - np.mean(series)
        cs = np.cumsum(ts)
        max_lag = min(len(ts) // 2, 50)
        rs = []
        for lag in range(2, max_lag):
            n = len(ts) // lag
            if n < 2:
                continue
            for i in range(n):
                seg = ts[i * lag:(i + 1) * lag]
                csw = cs[i * lag:(i + 1) * lag] - cs[i * lag]
                R = csw.max() - csw.min()
                S = seg.std(ddof=1)
                if S > 0:
                    rs.append((lag, R / S))
        if len(rs) < 10:
            return 0.5
        lags, vals = zip(*rs)
        h = np.polyfit(np.log(lags), np.log(vals), 1)[0]
        return max(0.1, min(0.9, h))