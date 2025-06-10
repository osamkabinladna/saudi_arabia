from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
from collections import deque


class AlgoEvent:
    # ───────────────────────── CONFIG ──────────────────────────
    def __init__(self):
        # base size for Gold leg
        self.base_lot = 2

        # timers & bookkeeping
        self.lasttradetime = datetime(1900, 1, 1)
        self.orderPairCnt = 0
        self.osOrder: dict = {}
        self.contractSize = {}
        self.minVolume = {}
        self.volumeIncrement = {}
        self.entry_time = {}

        # Enhanced regime detection parameters
        self.current_regime = "neutral"
        self.regime_confidence = 0.0
        self.regime_history = deque(maxlen=50)
        self.dynamic_thresholds = True

        # Regime detection windows
        self.short_window = 20
        self.medium_window = 50
        self.long_window = 100

        # lookback windows
        self.hurst_window = 200
        self.signal_window = 100
        self.mom_window = 100
        self.volatility_window = 30

        # base MR thresholds
        self.base_mr_hurst_thresh = 0.65
        self.base_z_entry_mr = 1.5
        self.base_z_exit_mr = 0.5
        self.max_hold_hours = 12

        # base TF thresholds
        self.base_tf_hurst_thresh = 0.35
        self.base_mom_entry_tf = 0.8
        self.base_mom_exit_tf = 0.3

        # initialize current thresholds to base
        self.mr_hurst_thresh = self.base_mr_hurst_thresh
        self.z_entry_mr = self.base_z_entry_mr
        self.z_exit_mr = self.base_z_exit_mr
        self.tf_hurst_thresh = self.base_tf_hurst_thresh
        self.mom_entry_tf = self.base_mom_entry_tf
        self.mom_exit_tf = self.base_mom_exit_tf

        # data buffers & counters
        self.arr_gold = []
        self.arr_silver = []
        self.arr_spread = []
        self.arr_filtered_spread = []
        self.arr_returns = []
        self.total_trades = 0
        self.mr_trades = 0
        self.tf_trades = 0

        # equity tracking
        self.current_equity = 0.0
        self.peak_equity = 0.0
        self.volatility_lookback = 20
        self.base_volatility = 0.02

        # Volatility tracking
        self.realized_vol = 0.02
        self.ewma_vol = 0.02
        self.vol_alpha = 0.94  # EWMA decay factor

        # Kalman filter state
        self.kf_initialized = False
        self.kf_state = np.array([0.0, 0.0])
        self.kf_covariance = np.eye(2) * 0.1
        self.kf_process_noise = 0.01
        self.kf_measurement_noise = 0.1

        # Dynamic hedge ratio parameters
        self.hedge_window = 60
        self.hedge_ratio_history = deque(maxlen=100)
        self.current_hedge_ratio = 1.0
        self.use_dynamic_hedge = True

    # ───────────────────────── STARTUP ─────────────────────────
    def start(self, mEvt):
        self.gold, self.silver = mEvt["subscribeList"]
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        for sym in (self.gold, self.silver):
            spec = self.evt.getContractSpec(sym)
            self.contractSize[sym] = spec["contractSize"]
            self.minVolume[sym] = spec.get("minVolume", 1)
            self.volumeIncrement[sym] = spec.get("volumeIncrement", 1)
        self.evt.consoleLog("Starting Enhanced Dual-Regime Strategy with Advanced Features")
        self.evt.consoleLog(f"Gold (Y): {self.gold}, Silver (X): {self.silver}")
        self.evt.start()

    # ───────────────────────── KALMAN FILTER ───────────────────
    def _update_kalman_filter(self, raw_spread):
        """Enhanced Kalman filter with adaptive noise parameters"""
        if not self.kf_initialized:
            self.kf_state[0] = raw_spread
            self.kf_initialized = True
            return raw_spread

        # Adaptive noise based on recent volatility
        if len(self.arr_returns) > 10:
            recent_vol = np.std(self.arr_returns[-10:])
            self.kf_measurement_noise = max(0.05, min(0.5, recent_vol * 5))

        F = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
        Q = np.eye(2) * self.kf_process_noise

        # Predict
        self.kf_state = F @ self.kf_state
        self.kf_covariance = F @ self.kf_covariance @ F.T + Q

        # Update
        H = np.array([[1.0, 0.0]])
        R = np.array([[self.kf_measurement_noise]])
        y = raw_spread - float(H @ self.kf_state)
        S = H @ self.kf_covariance @ H.T + R
        K = (self.kf_covariance @ H.T) / S.item()

        self.kf_state = self.kf_state + (K.flatten() * y)
        I_KH = np.eye(2) - K @ H
        self.kf_covariance = I_KH @ self.kf_covariance

        return float(self.kf_state[0])

    # ───────────────────────── ENHANCED REGIME DETECTION ───────────────────
    def _detect_regime_enhanced(self, hurst, momentum, volatility):
        """Enhanced regime detection using multiple indicators"""
        # Calculate regime scores
        mr_score = 0.0
        tf_score = 0.0

        # Hurst component
        if hurst < 0.45:
            mr_score += 0.4
        elif hurst > 0.55:
            tf_score += 0.4

        # Momentum component
        if abs(momentum) < 0.3:
            mr_score += 0.2
        elif abs(momentum) > 0.7:
            tf_score += 0.3

        # Volatility clustering
        if len(self.arr_returns) > 30:
            recent_vol = np.std(self.arr_returns[-10:])
            longer_vol = np.std(self.arr_returns[-30:])
            vol_ratio = recent_vol / (longer_vol + 1e-6)

            if vol_ratio > 1.5:  # High volatility clustering
                tf_score += 0.2
            elif vol_ratio < 0.7:  # Low volatility
                mr_score += 0.2

        # Multi-timeframe analysis
        if len(self.arr_filtered_spread) >= self.long_window:
            short_hurst = self._calc_hurst_corrected(self.arr_filtered_spread[-self.short_window:])
            med_hurst = self._calc_hurst_corrected(self.arr_filtered_spread[-self.medium_window:])

            if short_hurst < 0.4 and med_hurst < 0.45:
                mr_score += 0.2
            elif short_hurst > 0.6 and med_hurst > 0.55:
                tf_score += 0.2

        # Determine regime
        if mr_score > tf_score and mr_score > 0.5:
            regime = "mr"
            confidence = mr_score
        elif tf_score > mr_score and tf_score > 0.5:
            regime = "tf"
            confidence = tf_score
        else:
            regime = "neutral"
            confidence = 0.0

        return regime, confidence

    # ───────────────────────── DYNAMIC PARAMETERS ───────────────────
    def _calculate_market_volatility_enhanced(self):
        """Enhanced volatility calculation using multiple methods"""
        if len(self.arr_returns) < 20:
            return self.base_volatility

        # Realized volatility
        returns = np.array(self.arr_returns[-self.volatility_window:])
        self.realized_vol = np.std(returns) * np.sqrt(252 / 24)  # Annualized hourly vol

        # EWMA volatility
        if len(self.arr_returns) > 1:
            ret = self.arr_returns[-1]
            self.ewma_vol = np.sqrt(self.vol_alpha * self.ewma_vol ** 2 +
                                    (1 - self.vol_alpha) * ret ** 2)

        # GARCH-like conditional volatility
        if len(returns) > 5:
            squared_returns = returns ** 2
            garch_vol = np.sqrt(0.1 * np.mean(squared_returns) +
                                0.9 * squared_returns[-1])
        else:
            garch_vol = self.realized_vol

        # Combine volatility measures
        combined_vol = 0.5 * self.realized_vol + 0.3 * self.ewma_vol + 0.2 * garch_vol

        return max(0.001, combined_vol)

    def _update_dynamic_parameters(self):
        """Validated volatility-based parameter adjustment"""
        vol = self._calculate_market_volatility_enhanced()
        ratio = vol / self.base_volatility

        # Define volatility regimes with validation
        if ratio > 2.5:
            vol_regime = "extreme_high"
        elif ratio > 1.5:
            vol_regime = "high"
        elif ratio < 0.5:
            vol_regime = "low"
        elif ratio < 0.8:
            vol_regime = "normal_low"
        else:
            vol_regime = "normal"

        # MR adjustments with bounds
        if vol_regime == "extreme_high":
            self.z_entry_mr = min(3.0, self.base_z_entry_mr * 1.8)
            self.z_exit_mr = min(1.0, self.base_z_exit_mr * 1.5)
            self.mr_hurst_thresh = max(0.5, self.base_mr_hurst_thresh - 0.1)
        elif vol_regime == "high":
            self.z_entry_mr = min(2.5, self.base_z_entry_mr * 1.5)
            self.z_exit_mr = min(0.8, self.base_z_exit_mr * 1.3)
            self.mr_hurst_thresh = max(0.55, self.base_mr_hurst_thresh - 0.05)
        elif vol_regime == "low":
            self.z_entry_mr = max(1.0, self.base_z_entry_mr * 0.7)
            self.z_exit_mr = max(0.3, self.base_z_exit_mr * 0.8)
            self.mr_hurst_thresh = min(0.75, self.base_mr_hurst_thresh + 0.05)
        else:
            self.z_entry_mr = self.base_z_entry_mr
            self.z_exit_mr = self.base_z_exit_mr
            self.mr_hurst_thresh = self.base_mr_hurst_thresh

        # TF adjustments with bounds
        if vol_regime in ["extreme_high", "high"]:
            self.mom_entry_tf = min(1.5, self.base_mom_entry_tf * 1.4)
            self.mom_exit_tf = min(0.5, self.base_mom_exit_tf * 1.2)
            self.tf_hurst_thresh = min(0.45, self.base_tf_hurst_thresh + 0.05)
        elif vol_regime == "low":
            self.mom_entry_tf = max(0.5, self.base_mom_entry_tf * 0.8)
            self.mom_exit_tf = max(0.2, self.base_mom_exit_tf * 0.9)
            self.tf_hurst_thresh = max(0.25, self.base_tf_hurst_thresh - 0.05)
        else:
            self.mom_entry_tf = self.base_mom_entry_tf
            self.mom_exit_tf = self.base_mom_exit_tf
            self.tf_hurst_thresh = self.base_tf_hurst_thresh

    # ───────────────────────── HEDGE RATIO ───────────────────
    def _calculate_dynamic_hedge_ratio(self):
        """Calculate optimal hedge ratio using rolling regression"""
        if len(self.arr_gold) < self.hedge_window or len(self.arr_silver) < self.hedge_window:
            return 1.0

        # Get recent price data
        gold_prices = np.array(self.arr_gold[-self.hedge_window:])
        silver_prices = np.array(self.arr_silver[-self.hedge_window:])

        # Calculate returns
        gold_returns = np.diff(np.log(gold_prices))
        silver_returns = np.diff(np.log(silver_prices))

        if len(gold_returns) < 2:
            return 1.0

        # Multiple methods for robustness
        # 1. OLS regression
        try:
            slope, _, r_value, _, _ = stats.linregress(silver_returns, gold_returns)
            ols_ratio = slope
        except:
            ols_ratio = 1.0

        # 2. Variance ratio
        var_ratio = np.var(gold_returns) / (np.var(silver_returns) + 1e-10)
        std_ratio = np.sqrt(var_ratio)

        # 3. Correlation-adjusted ratio
        corr = np.corrcoef(gold_returns, silver_returns)[0, 1]
        corr_ratio = std_ratio * corr

        # Combine methods with weights
        hedge_ratio = 0.5 * ols_ratio + 0.3 * corr_ratio + 0.2 * std_ratio

        # Apply bounds and smoothing
        hedge_ratio = max(0.5, min(2.0, hedge_ratio))

        # Smooth with historical values
        self.hedge_ratio_history.append(hedge_ratio)
        if len(self.hedge_ratio_history) > 5:
            smoothed_ratio = np.mean(list(self.hedge_ratio_history)[-5:])
        else:
            smoothed_ratio = hedge_ratio

        return smoothed_ratio

    # ───────────────────────── DATA LOOP ───────────────────────
    def on_bulkdatafeed(self, isSync, bd, ab):
        if not isSync:
            return
        now = bd[self.gold]["timestamp"]
        if now < self.lasttradetime + timedelta(hours=1):
            return
        self.lasttradetime = now

        # receive prices
        p_g = bd[self.gold]["lastPrice"]
        p_s = bd[self.silver]["lastPrice"]
        self.arr_gold.append(p_g)
        self.arr_silver.append(p_s)
        if len(self.arr_gold) > self.hurst_window:
            self.arr_gold.pop(0)
            self.arr_silver.pop(0)
        if len(self.arr_gold) < self.hurst_window:
            return

        # spread & smoothing
        raw = np.log(p_g) - np.log(p_s)
        filt = self._update_kalman_filter(raw)
        self.arr_spread.append(raw)
        self.arr_filtered_spread.append(filt)

        # Calculate returns for volatility
        if len(self.arr_filtered_spread) > 1:
            ret = self.arr_filtered_spread[-1] - self.arr_filtered_spread[-2]
            self.arr_returns.append(ret)
            if len(self.arr_returns) > self.hurst_window:
                self.arr_returns.pop(0)

        if len(self.arr_filtered_spread) > self.hurst_window:
            self.arr_filtered_spread.pop(0)

        # Update dynamic hedge ratio
        if self.use_dynamic_hedge:
            self.current_hedge_ratio = self._calculate_dynamic_hedge_ratio()

        # dynamic thresholds
        if self.dynamic_thresholds:
            self._update_dynamic_parameters()

        # indicators
        h = self._calc_hurst_corrected(self.arr_filtered_spread[-self.hurst_window:])
        z = self._calc_zscore_robust(self.arr_filtered_spread[-self.signal_window:])
        m = self._calc_momentum_enhanced(self.arr_filtered_spread[-self.mom_window:])
        vol = self._calculate_market_volatility_enhanced()

        # Enhanced regime detection
        new_regime, confidence = self._detect_regime_enhanced(h, m, vol)
        self.regime_confidence = confidence

        # Regime persistence check
        self.regime_history.append(new_regime)
        if len(self.regime_history) >= 3:
            recent_regimes = list(self.regime_history)[-3:]
            if all(r == new_regime for r in recent_regimes) and new_regime != self.current_regime:
                self.current_regime = new_regime
                self.evt.consoleLog(f">>> REGIME → {new_regime} (confidence: {confidence:.2f})")

        # debug
        self.evt.consoleLog(
            f"[DBG] H={h:.3f} Z={z:.2f} M={m:.2f} Vol={vol:.4f} "
            f"Regime={self.current_regime} Conf={self.regime_confidence:.2f} "
            f"HedgeRatio={self.current_hedge_ratio:.3f}"
        )

        # entry logic with enhanced conditions
        if not self.matchPairTradeID():
            if self.current_regime == "mr" and self.regime_confidence > 0.6:
                if abs(z) > self.z_entry_mr:
                    # Additional filter: check if spread is not accelerating
                    if len(self.arr_returns) > 5:
                        recent_accel = np.mean(self.arr_returns[-5:])
                        if abs(recent_accel) < 0.001:  # Not accelerating
                            d = -1 if z > 0 else +1
                            self._open_pair(d, "mr", now)
            elif self.current_regime == "tf" and self.regime_confidence > 0.6:
                if m > self.mom_entry_tf:
                    # Additional filter: momentum should be accelerating
                    if len(self.arr_filtered_spread) > 10:
                        m_prev = self._calc_momentum_enhanced(self.arr_filtered_spread[-self.mom_window - 5:-5])
                        if m > m_prev:  # Momentum increasing
                            self._open_pair(+1, "tf", now)

        # exit logic
        for leg1, leg2 in self.matchPairTradeID().items():
            o1 = self.osOrder.get(leg1)
            if not o1:
                continue
            if self.current_regime == "mr":
                held = now - self.entry_time.get(o1['orderRef'], now)
                # Enhanced exit conditions
                exit_conditions = [
                    abs(z) < self.z_exit_mr,
                    held > timedelta(hours=self.max_hold_hours),
                    self.regime_confidence < 0.4,  # Regime weakening
                    vol > self.base_volatility * 3  # Extreme volatility
                ]
                if any(exit_conditions):
                    self.closeOrder(leg1)
                    self.closeOrder(leg2)
            elif self.current_regime == "tf":
                bs1 = o1['buysell']
                side = bs1 if o1['instrument'] == self.gold else -bs1
                # Enhanced exit conditions
                exit_conditions = [
                    (side == 1 and m < self.mom_exit_tf),
                    (side == -1 and m > -self.mom_exit_tf),
                    self.regime_confidence < 0.4,
                    abs(z) > 2.5  # Extreme reversion signal
                ]
                if any(exit_conditions):
                    self.closeOrder(leg1)
                    self.closeOrder(leg2)

        # summary
        open_ct = len(self.matchPairTradeID())
        self.evt.consoleLog(
            f"t={now:%Y-%m-%d %H:%M} H={h:.3f} Z={z:.2f} M={m:.2f} "
            f"Regime={self.current_regime} Open={open_ct} "
            f"Total={self.total_trades} MR={self.mr_trades} TF={self.tf_trades}"
        )

    # ─────────────────── ORDER HELPERS ───────────────────
    def _open_pair(self, direction, regime, now):
        self.orderPairCnt += 1
        self.total_trades += 1
        if regime == "mr":
            self.mr_trades += 1
        else:
            self.tf_trades += 1

        inc_g, mv_g = self.volumeIncrement[self.gold], self.minVolume[self.gold]
        vol_g = max(mv_g, round(self.base_lot / inc_g) * inc_g)

        p_g, p_s = self.arr_gold[-1], self.arr_silver[-1]
        cs_g, cs_s = self.contractSize[self.gold], self.contractSize[self.silver]

        # Apply dynamic hedge ratio
        not_g = vol_g * p_g * cs_g
        raw_s = not_g / (p_s * cs_s * self.current_hedge_ratio)

        inc_s, mv_s = self.volumeIncrement[self.silver], self.minVolume[self.silver]
        vol_s = max(mv_s, round(raw_s / inc_s) * inc_s)

        self.entry_time[self.orderPairCnt] = now
        self.openOrder(direction, self.gold, self.orderPairCnt, vol_g)
        self.openOrder(-direction, self.silver, self.orderPairCnt, vol_s)
        self.evt.consoleLog(
            f">>> OPEN {regime.upper()} dir={direction:+d} G={vol_g} S={vol_s} "
            f"HedgeRatio={self.current_hedge_ratio:.3f}"
        )

    def matchPairTradeID(self):
        pairs = {}
        for tid, od in self.osOrder.items():
            for tid2, od2 in self.osOrder.items():
                if tid != tid2 and od['orderRef'] == od2['orderRef'] and tid not in pairs:
                    pairs[tid] = tid2
                    break
        return pairs

    def closeOrder(self, tradeID):
        self.evt.sendOrder(AlgoAPIUtil.OrderObject(tradeID=tradeID, openclose='close'))

    def openOrder(self, bs, inst, ref, vol):
        self.evt.sendOrder(AlgoAPIUtil.OrderObject(
            instrument=inst, orderRef=ref, volume=vol,
            openclose='open', buysell=bs, ordertype=0))

    # ───────────────── required stubs ─────────────────
    def on_marketdatafeed(self, md, ab):
        pass

    def on_orderfeed(self, of):
        pass

    def on_dailyPLfeed(self, pl):
        if 'totalPL' in pl:
            self.current_equity = pl['totalPL']
            self.peak_equity = max(self.peak_equity, self.current_equity)

    def on_openPositionfeed(self, op, oo, uo):
        self.osOrder = oo

    def on_dailyDumpData(self, dump):
        pass

    # ───────────────── ENHANCED INDICATORS ─────────────────
    def _calc_zscore_robust(self, series):
        """Robust Z-score calculation with outlier handling"""
        vals = []
        for x in series:
            try:
                vals.append(float(x))
            except:
                continue
        if len(vals) < 10:
            return 0.0

        arr = np.array(vals)

        # Use robust statistics (median and MAD)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))

        # Traditional mean/std as fallback
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)

        if mad > 0:
            # Robust z-score
            z_robust = (arr[-1] - median) / (1.4826 * mad)
        else:
            # Fall back to traditional z-score
            z_robust = (arr[-1] - mean) / std if std > 0 else 0.0

        # Clip extreme values
        return np.clip(z_robust, -4, 4)

    def _calc_momentum_enhanced(self, series):
        """Enhanced momentum with multiple timeframes"""
        vals = []
        for x in series:
            try:
                vals.append(float(x))
            except:
                continue
        if len(vals) < 10:
            return 0.0

        # Calculate returns
        rets = np.diff(vals) / np.abs(vals[:-1] + 1e-10)

        if len(rets) < 5:
            return 0.0

        # Short-term momentum
        short_mom = np.mean(rets[-5:])

        # Medium-term momentum
        if len(rets) >= 20:
            med_mom = np.mean(rets[-20:])
        else:
            med_mom = np.mean(rets)

        # Momentum quality (consistency)
        if len(rets) > 10:
            rolling_means = [np.mean(rets[i:i + 5]) for i in range(len(rets) - 5)]
            mom_consistency = 1 - np.std(rolling_means) / (np.abs(np.mean(rolling_means)) + 1e-10)
        else:
            mom_consistency = 0.5

        # Combine momentum signals
        combined_mom = 0.6 * short_mom + 0.3 * med_mom + 0.1 * mom_consistency

        # Normalize by volatility
        vol = np.std(rets) if len(rets) > 1 else 1.0
        return combined_mom / (vol + 1e-10) if vol > 0 else 0.0

    def _calc_hurst_corrected(self, series):
        """Corrected Hurst exponent calculation"""
        vals = []
        for x in series:
            try:
                vals.append(float(x))
            except:
                continue
        N = len(vals)
        if N < 20:
            return 0.5

        # Detrend the series
        ts = np.array(vals)
        trend = np.polyfit(range(N), ts, 1)
        detrended = ts - (trend[0] * np.arange(N) + trend[1])

        # Calculate cumulative sum
        mean_centered = detrended - np.mean(detrended)
        cumsum = np.cumsum(mean_centered)

        # R/S analysis
        lags = []
        rs_values = []

        min_lag = 4
        max_lag = min(N // 4, 100)

        for lag in range(min_lag, max_lag, 2):
            if lag >= N:
                continue

            # Split into non-overlapping subseries
            n_segments = N // lag
            if n_segments < 2:
                continue

            rs_list = []
            for i in range(n_segments):
                start = i * lag
                end = start + lag
                if end > N:
                    break

                segment = mean_centered[start:end]
                segment_cumsum = np.cumsum(segment)

                R = np.max(segment_cumsum) - np.min(segment_cumsum)
                S = np.std(segment, ddof=1)

                if S > 1e-10:
                    rs_list.append(R / S)

            if rs_list:
                lags.append(lag)
                rs_values.append(np.mean(rs_list))

        if len(lags) < 5:
            return 0.5

        # Log-log regression
        log_lags = np.log(lags)
        log_rs = np.log(rs_values)

        # Weighted regression (give more weight to smaller lags)
        weights = 1.0 / np.sqrt(lags)
        coef = np.polyfit(log_lags, log_rs, 1, w=weights)
        hurst = coef[0]

        # Anis-Lloyd correction for small samples
        if N < 100:
            expected_h = 0.5 + (0.72 / np.sqrt(N))
            hurst = hurst - (expected_h - 0.5)

        return max(0.1, min(0.9, hurst))