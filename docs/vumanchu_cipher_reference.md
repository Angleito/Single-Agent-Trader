# VuManChu Cipher Reference - Full Pine Script Source

*Date: 2025-06-11*

This document contains the complete Pine Script source code for both VuManChu Cipher A and Cipher B indicators. These are the original open-source implementations that our AI Trading Bot will re-implement in Python for local control.

## Cipher A

```pine
1	//@version=4
2	
3	// Thanks Dynausmaux for the code
4	// Based on Cipher_A from falconCoin https://www.tradingview.com/script/cAw5GEAB-Market-Cipher-A-free-version-1-1/
5	// Thanks to LazyBear foor WaveTrend Oscillator https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/
6	// I just added the red diamond, blood diamond and yellowX pattern, i dont know if is exact but seems to be.
7	// Still need a lot of visual adjustments to look like market cipher A but it's an attempt
8	study(title="VuManChu Cipher A", shorttitle="VMC Cipher_A", overlay=true)
9	
10	// FUNCTIONS {
11	
12	// WaveTrend
13	f_wavetrend(_src, _chlen, _avg, _malen) =>
14	    _esa = ema(_src, _chlen)
15	    _de = ema(abs(_src - _esa), _chlen)
16	    _ci = (_src - _esa) / (0.015 * _de)
17	    _tci = ema(_ci, _avg)
18	    _wt1 = _tci
19	    _wt2 = sma(_wt1, _malen)
20	    [_wt1, _wt2]
21	
22	// 8 EMA Ribbon
23	f_emaRibbon(_src, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8) =>
24	    _ema1 = ema(_src, _e1)
25	    _ema2 = ema(_src, _e2)
26	    _ema3 = ema(_src, _e3)
27	    _ema4 = ema(_src, _e4)
28	    _ema5 = ema(_src, _e5)
29	    _ema6 = ema(_src, _e6)
30	    _ema7 = ema(_src, _e7)
31	    _ema8 = ema(_src, _e8)
32	    [_ema1, _ema2, _ema3, _ema4, _ema5, _ema6, _ema7, _ema8]
33	
34	f_rsimfi(_period, _multiplier, _tf) => security(syminfo.tickerid, _tf, sma(((close - open) / (high - low)) * _multiplier, _period))
35	
36	// } FUNCTIONS
37	
38	// PARAMETERS {
39	
40	// WaveTrend
41	wtChannelLen = input(9, title = 'WT Channel Length')
42	wtAverageLen = input(13, title = 'WT Average Length')
43	wtMASource = input(hlc3, title = 'WT MA Source')
44	wtMALen = input(3, title = 'WT MA Length')
45	
46	// WaveTrend Overbought & Oversold lines
47	obLevel = input(53, title = 'WT Overbought Level 1')
48	obLevel2 = input(60, title = 'WT Overbought Level 2')
49	obLevel3 = input(100, title = 'WT Overbought Level 3')
50	osLevel = input(-53, title = 'WT Oversold Level 1')
51	osLevel2 = input(-60, title = 'WT Oversold Level 2')
52	osLevel3 = input(-80, title = 'WT Oversold Level 3')
53	
54	// EMA Ribbon
55	showRibbon = input(true, "Show Ribbon")
56	ema1Len = input(5, title = "EMA 1 Length")
57	ema2Len = input(11, title = "EMA 2 Length")
58	ema3Len = input(15, title = "EMA 3 Length")
59	ema4Len = input(18, title = "EMA 4 Length")
60	ema5Len = input(21, title = "EMA 5 Length")
61	ema6Len = input(24, title = "EMA 6 Length")
62	ema7Len = input(28, title = "EMA 7 Length")
63	ema8Len = input(34, title = "EMA 8 Length")
64	
65	// RSI
66	rsiSRC = input(close, title = "RSI Source")
67	rsiLen = input(14, title = "RSI Length")
68	rsiOversold = input(30, title = 'RSI Oversold', minval = 50, maxval = 100)
69	rsiOverbought = input(60, title = 'RSI Overbought', minval = 0, maxval = 50)
70	
71	// RSI+MFI
72	rsiMFIShow = input(true, title = "Show RSI+MFI")
73	rsiMFIperiod = input(60, title = 'RSI+MFI Period')
74	rsiMFIMultiplier = input(150, title = 'RSI+MFI Area multiplier')
75	
76	// }
77	
78	// CALCULATE INDICATORS {
79	
80	// EMA Ribbon
81	[ema1, ema2, ema3, ema4, ema5, ema6, ema7, ema8] = f_emaRibbon(close, ema1Len, ema2Len, ema3Len, ema4Len, ema5Len, ema6Len, ema7Len, ema8Len)
82	
83	// RSI
84	rsi = rsi(rsiSRC, rsiLen)
85	
86	// Calculates WaveTrend
87	[wt1, wt2] = f_wavetrend(wtMASource, wtChannelLen, wtAverageLen, wtMALen)
88	
89	// WaveTrend Conditions
90	wtOverSold = wt2 <= osLevel
91	wtOverBought = wt2 >= obLevel
92	wtGreenCross = cross(wt1, wt2)
93	wtRedCross = cross(wt2, wt1)
94	wtCrossUp = wt2 - wt1 <= 0
95	wtCrossDown = wt2 - wt1 >= 0
96	
97	// RSI + MFI
98	rsiMFI = f_rsimfi(rsiMFIperiod, rsiMFIMultiplier, timeframe.period)
99	
100	// Signals
101	longEma = crossover(ema2, ema8)
102	shortEma = crossover(ema8, ema2)
103	redCross = crossunder(ema1, ema2)
104	greenCross = crossunder(ema2, ema1)
105	blueTriangleUp = crossover(ema2, ema3)
106	blueTriangleDown = crossover(ema3, ema2)
107	redDiamond = wtGreenCross and wtCrossDown
108	greenDiamond = wtRedCross and wtCrossUp
109	yellowCrossUp = redDiamond and wt2 < 45 and wt2 > osLevel3 and rsi < 30 and rsi > 15 //and rsiMFI < -5
110	yellowCrossDown = greenDiamond and wt2 > 55 and wt2 < obLevel3 and rsi < 70 and rsi < 85 //and rsiMFI > 95
111	dumpDiamond = redDiamond and redCross
112	moonDiamond = greenDiamond and greenCross
113	bullCandle = open > ema2 and open > ema8 and (close[1] > open[1]) and (close > open) and not redDiamond and not redCross
114	bearCandle = open < ema2 and open < ema8 and (close[1] < open[1]) and (close < open) and not greenDiamond and not redCross
115	
116	// } CALCULATE INDICATORS
117	
118	// DRAW {
119	
120	// EMA Ribbon
121	ribbonDir = ema8 < ema2
122	colorEma = ribbonDir ? color.green : color.red
123	p1 = plot(ema1, color=showRibbon ? ribbonDir ? #1573d4 : color.gray : na, linewidth=1, transp=15, title="EMA 1")
124	p2 = plot(ema2, color=showRibbon ? ribbonDir ? #3096ff : color.gray : na, linewidth=1, transp=15, title="EMA 2")
125	plot(ema3, color=showRibbon ? ribbonDir ? #57abff : color.gray : na, linewidth=1, transp=15, title="EMA 3")
126	plot(ema4, color=showRibbon ? ribbonDir ? #85c2ff : color.gray : na, linewidth=1, transp=15, title="EMA 4")
127	plot(ema5, color=showRibbon ? ribbonDir ? #9bcdff : color.gray : na, linewidth=1, transp=15, title="EMA 5")
128	plot(ema6, color=showRibbon ? ribbonDir ? #b3d9ff : color.gray : na, linewidth=1, transp=15, title="EMA 6")
129	plot(ema7, color=showRibbon ? ribbonDir ? #c9e5ff : color.gray : na, linewidth=1, transp=15, title="EMA 7")
130	plot(ema8, color=showRibbon ? ribbonDir ? #dfecfb : color.gray : na, linewidth=1, transp=15, title="EMA 8")
131	p8 = plot(ema8, color=showRibbon ? na : colorEma, linewidth=1, transp=0, title="EMA 8")
132	fill(p1, p2, color = #1573d4, transp = 85)
133	fill(p2, p8, color = #363a45, transp = 85)
134	
135	// SHAPES
136	
137	plotshape(longEma, style=shape.circle, color=#009688, location=location.abovebar, size=size.tiny, title="Long EMA Signal", transp=50)
138	plotshape(shortEma, style=shape.circle, color=#f44336, location=location.abovebar, size=size.tiny, title="Short EMA Signal", transp=50)
139	plotshape(redCross, style=shape.xcross, color=#ef5350, location=location.abovebar, size=size.tiny, title="Red Cross", transp=50, display=display.none)
140	plotshape(greenCross, style=shape.xcross, color=#66bb6a, location=location.abovebar, size=size.tiny, title="Green Cross", transp=50, display=display.none)
141	plotshape(blueTriangleUp, style=shape.triangleup, color=#0064ff, location=location.belowbar, size=size.tiny, title="Blue Triangle Bull", transp=50)
142	plotshape(blueTriangleDown, style=shape.triangledown, color=#0064ff, location=location.belowbar, size=size.tiny, title="Blue Triangle Bear", transp=50)
143	plotshape(redDiamond, style=shape.diamond, color=#ef9a9a, location=location.abovebar, size=size.tiny, title="Red Diamond", transp=25, display=display.none)
144	plotshape(greenDiamond, style=shape.diamond, color=#a5d6a7, location=location.belowbar, size=size.tiny, title="Green Diamond", transp=25, display=display.none)
145	plotshape(bullCandle, style=shape.diamond, color=#00796b, location=location.abovebar, size=size.tiny, title="Bull Candle", transp=75, display=display.none)
146	plotshape(bearCandle, style=shape.diamond, color=#d32f2f, location=location.belowbar, size=size.tiny, title="Bear Candle", transp=75, display=display.none)
147	plotshape(dumpDiamond, style=shape.diamond, color=#b71c1c, location=location.abovebar, size=size.small, title="Dump Diamond", transp=15, display=display.none)
148	plotshape(moonDiamond, style=shape.diamond, color=#1b5e20, location=location.belowbar, size=size.small, title="Moon Diamond", transp=15, display=display.none)
149	plotshape(yellowCrossUp, style=shape.xcross, color=#fbc02d, location=location.abovebar, size=size.small, title="Yellow Cross Up", transp=25)
150	plotshape(yellowCrossDown, style=shape.xcross, color=#fbc02d, location=location.belowbar, size=size.small, title="Yellow Cross Down", transp=25)
151	// } DRAW
152	
153	
154	// ALERTS {
155	alertcondition(condition=redDiamond, title="Red Diamond", message="Red Diamond")
156	alertcondition(condition=dumpDiamond, title="Dump Diamond", message="Dump Diamond")
157	alertcondition(condition=blueTriangleUp or blueTriangleDown, title="Blue Triangle", message="Blue Triangle")
158	alertcondition(condition=yellowCrossUp, title="Yellow X", message="Yellow X")
159	alertcondition(condition=redCross, title="Red X", message="Red X")
160	alertcondition(condition=greenCross, title="Green X", message="Green X")
161	alertcondition(condition=shortEma, title="Short Signal", message="Short Signal")
162	alertcondition(condition=longEma, title="Long Signal", message="Long Signal")
163	alertcondition(condition=shortEma or longEma, title="Long/Short Signal", message="Long/Short Signal")
164	// } ALERTS
```

## Cipher B

```pine
1	// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
2	// © vumanchu
3	
4	//@version=4
5	
6	//  Thanks to dynausmaux for the code
7	//  Thanks to falconCoin for https://www.tradingview.com/script/KVfgBvDd-Market-Cipher-B-Free-version-with-Buy-and-sell/ inspired me to start this.
8	//  Thanks to LazyBear for WaveTrend Oscillator https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/
9	//  Thanks to RicardoSantos for https://www.tradingview.com/script/3oeDh0Yq-RS-Price-Divergence-Detector-V2/
10	//  Thanks to LucemAnb for Plain Stochastic Divergence https://www.tradingview.com/script/FCUgF8ag-Plain-Stochastic-Divergence/
11	//  Thanks to andreholanda73 for MFI+RSI Area https://www.tradingview.com/script/UlGZzUAr/
12	//  I especially want to thank TradingView for its platform that facilitates development and learning.
13	
14	//
15	//  CIRCLES & TRIANGLES:
16	//    - LITTLE CIRCLE: They appear at all WaveTrend wave crossings.
17	//    - GREEN CIRCLE: The wavetrend waves are at the oversold level and have crossed up (bullish).
18	//    - RED CIRCLE: The wavetrend waves are at the overbought level and have crossed down (bearish).
19	//    - GOLD/ORANGE CIRCLE: When RSI is below 20, WaveTrend waves are below or equal to -80 and have crossed up after good bullish divergence (DONT BUY WHEN GOLD CIRCLE APPEAR).
20	//    - None of these circles are certain signs to trade. It is only information that can help you.
21	//    - PURPLE TRIANGLE: Appear when a bullish or bearish divergence is formed and WaveTrend waves crosses at overbought and oversold points.
22	//
23	//  NOTES:
24	//    - I am not an expert trader or know how to program pine script as such, in fact it is my first indicator only to study and all the code is copied and modified from other codes that are published in TradingView.
25	//    - I am very grateful to the entire TV community that publishes codes so that other newbies like me can learn and present their results. This is an attempt to imitate Market Cipher B.
26	//    - Settings by default are for 4h timeframe, divergences are more stronger and accurate. Haven't tested in all timeframes, only 2h and 4h.
27	//    - If you get an interesting result in other timeframes I would be very grateful if you would comment your configuration to implement it or at least check it.
28	//
29	//  CONTRIBUTIONS:
30	//    - Tip/Idea: Add higher timeframe analysis for bearish/bullish patterns at the current timeframe.
31	//    + Bearish/Bullish FLAG:
32	//      - MFI+RSI Area are RED (Below 0).
33	//      - Wavetrend waves are above 0 and crosses down.
34	//      - VWAP Area are below 0 on higher timeframe.
35	//      - This pattern reversed becomes bullish.
36	//    - Tip/Idea: Check the last heikinashi candle from 2 higher timeframe
37	//    + Bearish/Bullish DIAMOND:
38	//      - HT Candle is red
39	//      - WT > 0 and crossed down
40	
41	study(title = 'VuManChu B Divergences', shorttitle = 'VMC Cipher_B_Divergences')
42	
43	// PARAMETERS {
44	
45	// WaveTrend
46	wtShow = input(true, title = 'Show WaveTrend', type = input.bool)
47	wtBuyShow = input(true, title = 'Show Buy dots', type = input.bool)
48	wtGoldShow = input(true, title = 'Show Gold dots', type = input.bool)
49	wtSellShow = input(true, title = 'Show Sell dots', type = input.bool)
50	wtDivShow = input(true, title = 'Show Div. dots', type = input.bool)
51	vwapShow = input(true, title = 'Show Fast WT', type = input.bool)
52	wtChannelLen = input(9, title = 'WT Channel Length', type = input.integer)
53	wtAverageLen = input(12, title = 'WT Average Length', type = input.integer)
54	wtMASource = input(hlc3, title = 'WT MA Source', type = input.source)
55	wtMALen = input(3, title = 'WT MA Length', type = input.integer)
56	
57	// WaveTrend Overbought & Oversold lines
58	obLevel = input(53, title = 'WT Overbought Level 1', type = input.integer)
59	obLevel2 = input(60, title = 'WT Overbought Level 2', type = input.integer)
60	obLevel3 = input(100, title = 'WT Overbought Level 3', type = input.integer)
61	osLevel = input(-53, title = 'WT Oversold Level 1', type = input.integer)
62	osLevel2 = input(-60, title = 'WT Oversold Level 2', type = input.integer)
63	osLevel3 = input(-75, title = 'WT Oversold Level 3', type = input.integer)
64	
65	// Divergence WT
66	wtShowDiv = input(true, title = 'Show WT Regular Divergences', type = input.bool)
67	wtShowHiddenDiv = input(false, title = 'Show WT Hidden Divergences', type = input.bool)
68	showHiddenDiv_nl = input(true, title = 'Not apply OB/OS Limits on Hidden Divergences', type = input.bool)
69	wtDivOBLevel = input(45, title = 'WT Bearish Divergence min', type = input.integer)
70	wtDivOSLevel = input(-65, title = 'WT Bullish Divergence min', type = input.integer)
71	
72	// Divergence extra range
73	wtDivOBLevel_addshow = input(true, title = 'Show 2nd WT Regular Divergences', type = input.bool)
74	wtDivOBLevel_add = input(15, title = 'WT 2nd Bearish Divergence', type = input.integer)
75	wtDivOSLevel_add = input(-40, title = 'WT 2nd Bullish Divergence 15 min', type = input.integer)
76	
77	// RSI+MFI
78	rsiMFIShow = input(true, title = 'Show MFI', type = input.bool)
79	rsiMFIperiod = input(60,title = 'MFI Period', type = input.integer)
80	rsiMFIMultiplier = input(150, title = 'MFI Area multiplier', type = input.float)
81	rsiMFIPosY = input(2.5, title = 'MFI Area Y Pos', type = input.float)
82	
83	// RSI
84	rsiShow = input(false, title = 'Show RSI', type = input.bool)
85	rsiSRC = input(close, title = 'RSI Source', type = input.source)
86	rsiLen = input(14, title = 'RSI Length', type = input.integer)
87	rsiOversold = input(30, title = 'RSI Oversold', minval = 50, maxval = 100, type = input.integer)
88	rsiOverbought = input(60, title = 'RSI Overbought', minval = 0, maxval = 50, type = input.integer)
89	
90	// Divergence RSI
91	rsiShowDiv = input(false, title = 'Show RSI Regular Divergences', type = input.bool)
92	rsiShowHiddenDiv = input(false, title = 'Show RSI Hidden Divergences', type = input.bool)
93	rsiDivOBLevel = input(60, title = 'RSI Bearish Divergence min', type = input.integer)
94	rsiDivOSLevel = input(30, title = 'RSI Bullish Divergence min', type = input.integer)
95	
96	// RSI Stochastic
97	stochShow = input(true, title = 'Show Stochastic RSI', type = input.bool)
98	stochUseLog = input(true, title=' Use Log?', type = input.bool)
99	stochAvg = input(false, title='Use Average of both K & D', type = input.bool)
100	stochSRC = input(close, title = 'Stochastic RSI Source', type = input.source)
101	stochLen = input(14, title = 'Stochastic RSI Length', type = input.integer)
102	stochRsiLen = input(14, title = 'RSI Length ', type = input.integer)
103	stochKSmooth = input(3, title = 'Stochastic RSI K Smooth', type = input.integer)
104	stochDSmooth = input(3, title = 'Stochastic RSI D Smooth', type = input.integer)
105	
106	// Divergence stoch
107	stochShowDiv = input(false, title = 'Show Stoch Regular Divergences', type = input.bool)
108	stochShowHiddenDiv = input(false, title = 'Show Stoch Hidden Divergences', type = input.bool)
109	
110	// Schaff Trend Cycle
111	tcLine = input(false, title="Show Schaff TC line", type=input.bool)
112	tcSRC = input(close, title = 'Schaff TC Source', type = input.source)
113	tclength = input(10, title="Schaff TC", type=input.integer)
114	tcfastLength = input(23, title="Schaff TC Fast Lenght", type=input.integer)
115	tcslowLength = input(50, title="Schaff TC Slow Length", type=input.integer)
116	tcfactor = input(0.5, title="Schaff TC Factor", type=input.float)
117	
118	// Sommi Flag
119	sommiFlagShow = input(false, title = 'Show Sommi flag', type = input.bool)
120	sommiShowVwap = input(false, title = 'Show Sommi F. Wave', type = input.bool)
121	sommiVwapTF = input('720', title = 'Sommi F. Wave timeframe', type = input.string)
122	sommiVwapBearLevel = input(0, title = 'F. Wave Bear Level (less than)', type = input.integer)
123	sommiVwapBullLevel = input(0, title = 'F. Wave Bull Level (more than)', type = input.integer)
124	soomiFlagWTBearLevel = input(0, title = 'WT Bear Level (more than)', type = input.integer)
125	soomiFlagWTBullLevel = input(0, title = 'WT Bull Level (less than)', type = input.integer)
126	soomiRSIMFIBearLevel = input(0, title = 'Money flow Bear Level (less than)', type = input.integer)
127	soomiRSIMFIBullLevel = input(0, title = 'Money flow Bull Level (more than)', type = input.integer)
128	
129	// Sommi Diamond
130	sommiDiamondShow = input(false, title = 'Show Sommi diamond', type = input.bool)
131	sommiHTCRes = input('60', title = 'HTF Candle Res. 1', type = input.string)
132	sommiHTCRes2 = input('240', title = 'HTF Candle Res. 2', type = input.string)
133	soomiDiamondWTBearLevel = input(0, title = 'WT Bear Level (More than)', type = input.integer)
134	soomiDiamondWTBullLevel = input(0, title = 'WT Bull Level (Less than)', type = input.integer)
135	
136	// macd Colors
137	macdWTColorsShow = input(false, title = 'Show MACD Colors', type = input.bool)
138	macdWTColorsTF = input('240', title = 'MACD Colors MACD TF', type = input.string)
139	
140	darkMode = input(false, title = 'Dark mode', type = input.bool)
141	
142	// Colors
143	colorRed = #ff0000
144	colorPurple = #e600e6
145	colorGreen = #3fff00
146	colorOrange = #e2a400
147	colorYellow = #ffe500
148	colorWhite = #ffffff
149	colorPink = #ff00f0
150	colorBluelight = #31c0ff
151	
152	colorWT1 = #90caf9
153	colorWT2 = #0d47a1
154	
155	colorWT2_ = #131722
156	
157	colormacdWT1a = #4caf58
158	colormacdWT1b = #af4c4c
159	colormacdWT1c = #7ee57e
160	colormacdWT1d = #ff3535
161	
162	colormacdWT2a = #305630
163	colormacdWT2b = #310101
164	colormacdWT2c = #132213
165	colormacdWT2d = #770000
166	
167	// } PARAMETERS
168	
169	// FUNCTIONS {
170	
171	// Divergences
172	f_top_fractal(src) => src[4] < src[2] and src[3] < src[2] and src[2] > src[1] and src[2] > src[0]
173	f_bot_fractal(src) => src[4] > src[2] and src[3] > src[2] and src[2] < src[1] and src[2] < src[0]
174	f_fractalize(src) => f_top_fractal(src) ? 1 : f_bot_fractal(src) ? -1 : 0
175	
176	f_findDivs(src, topLimit, botLimit, useLimits) =>
177	    fractalTop = f_fractalize(src) > 0 and (useLimits ? src[2] >= topLimit : true) ? src[2] : na
178	    fractalBot = f_fractalize(src) < 0 and (useLimits ? src[2] <= botLimit : true) ? src[2] : na
179	    highPrev = valuewhen(fractalTop, src[2], 0)[2]
180	    highPrice = valuewhen(fractalTop, high[2], 0)[2]
181	    lowPrev = valuewhen(fractalBot, src[2], 0)[2]
182	    lowPrice = valuewhen(fractalBot, low[2], 0)[2]
183	    bearSignal = fractalTop and high[2] > highPrice and src[2] < highPrev
184	    bullSignal = fractalBot and low[2] < lowPrice and src[2] > lowPrev
185	    bearDivHidden = fractalTop and high[2] < highPrice and src[2] > highPrev
186	    bullDivHidden = fractalBot and low[2] > lowPrice and src[2] < lowPrev
187	    [fractalTop, fractalBot, lowPrev, bearSignal, bullSignal, bearDivHidden, bullDivHidden]
188	
189	// RSI+MFI
190	f_rsimfi(_period, _multiplier, _tf) => security(syminfo.tickerid, _tf, sma(((close - open) / (high - low)) * _multiplier, _period) - rsiMFIPosY)
191	
192	// WaveTrend
193	f_wavetrend(src, chlen, avg, malen, tf) =>
194	    tfsrc = security(syminfo.tickerid, tf, src)
195	    esa = ema(tfsrc, chlen)
196	    de = ema(abs(tfsrc - esa), chlen)
197	    ci = (tfsrc - esa) / (0.015 * de)
198	    wt1 = security(syminfo.tickerid, tf, ema(ci, avg))
199	    wt2 = security(syminfo.tickerid, tf, sma(wt1, malen))
200	    wtVwap = wt1 - wt2
201	    wtOversold = wt2 <= osLevel
202	    wtOverbought = wt2 >= obLevel
203	    wtCross = cross(wt1, wt2)
204	    wtCrossUp = wt2 - wt1 <= 0
205	    wtCrossDown = wt2 - wt1 >= 0
206	    wtCrosslast = cross(wt1[2], wt2[2])
207	    wtCrossUplast = wt2[2] - wt1[2] <= 0
208	    wtCrossDownlast = wt2[2] - wt1[2] >= 0
209	    [wt1, wt2, wtOversold, wtOverbought, wtCross, wtCrossUp, wtCrossDown, wtCrosslast, wtCrossUplast, wtCrossDownlast, wtVwap]
210	
211	// Schaff Trend Cycle
212	f_tc(src, length, fastLength, slowLength) =>
213	    ema1 = ema(src, fastLength)
214	    ema2 = ema(src, slowLength)
215	    macdVal = ema1 - ema2
216	    alpha = lowest(macdVal, length)
217	    beta = highest(macdVal, length) - alpha
218	    gamma = (macdVal - alpha) / beta * 100
219	    gamma := beta > 0 ? gamma : nz(gamma[1])
220	    delta = gamma
221	    delta := na(delta[1]) ? delta : delta[1] + tcfactor * (gamma - delta[1])
222	    epsilon = lowest(delta, length)
223	    zeta = highest(delta, length) - epsilon
224	    eta = (delta - epsilon) / zeta * 100
225	    eta := zeta > 0 ? eta : nz(eta[1])
226	    stcReturn = eta
227	    stcReturn := na(stcReturn[1]) ? stcReturn : stcReturn[1] + tcfactor * (eta - stcReturn[1])
228	    stcReturn
229	
230	// Stochastic RSI
231	f_stochrsi(_src, _stochlen, _rsilen, _smoothk, _smoothd, _log, _avg) =>
232	    src = _log ? log(_src) : _src
233	    rsi = rsi(src, _rsilen)
234	    kk = sma(stoch(rsi, rsi, rsi, _stochlen), _smoothk)
235	    d1 = sma(kk, _smoothd)
236	    avg_1 = avg(kk, d1)
237	    k = _avg ? avg_1 : kk
238	    [k, d1]
239	
240	// MACD
241	f_macd(src, fastlen, slowlen, sigsmooth, tf) =>
242	    fast_ma = security(syminfo.tickerid, tf, ema(src, fastlen))
243	    slow_ma = security(syminfo.tickerid, tf, ema(src, slowlen))
244	    macd = fast_ma - slow_ma,
245	    signal = security(syminfo.tickerid, tf, sma(macd, sigsmooth))
246	    hist = macd - signal
247	    [macd, signal, hist]
248	
249	// MACD Colors on WT
250	f_macdWTColors(tf) =>
251	    hrsimfi = f_rsimfi(rsiMFIperiod, rsiMFIMultiplier, tf)
252	    [macd, signal, hist] = f_macd(close, 28, 42, 9, macdWTColorsTF)
253	    macdup = macd >= signal
254	    macddown = macd <= signal
255	    macdWT1Color = macdup ? hrsimfi > 0 ? colormacdWT1c : colormacdWT1a : macddown ? hrsimfi < 0 ? colormacdWT1d : colormacdWT1b : na
256	    macdWT2Color = macdup ? hrsimfi < 0 ? colormacdWT2c : colormacdWT2a : macddown ? hrsimfi < 0 ? colormacdWT2d : colormacdWT2b : na
257	    [macdWT1Color, macdWT2Color]
258	
259	// Get higher timeframe candle
260	f_getTFCandle(_tf) =>
261	    _open  = security(heikinashi(syminfo.tickerid), _tf, open, barmerge.gaps_off, barmerge.lookahead_on)
262	    _close = security(heikinashi(syminfo.tickerid), _tf, close, barmerge.gaps_off, barmerge.lookahead_on)
263	    _high  = security(heikinashi(syminfo.tickerid), _tf, high, barmerge.gaps_off, barmerge.lookahead_on)
264	    _low   = security(heikinashi(syminfo.tickerid), _tf, low, barmerge.gaps_off, barmerge.lookahead_on)
265	    hl2   = (_high + _low) / 2.0
266	    newBar = change(_open)
267	    candleBodyDir = _close > _open
268	    [candleBodyDir, newBar]
269	
270	// Sommi flag
271	f_findSommiFlag(tf, wt1, wt2, rsimfi, wtCross, wtCrossUp, wtCrossDown) =>
272	    [hwt1, hwt2, hwtOversold, hwtOverbought, hwtCross, hwtCrossUp, hwtCrossDown, hwtCrosslast, hwtCrossUplast, hwtCrossDownlast, hwtVwap] = f_wavetrend(wtMASource, wtChannelLen, wtAverageLen, wtMALen, tf)
273	
274	    bearPattern = rsimfi < soomiRSIMFIBearLevel and
275	                   wt2 > soomiFlagWTBearLevel and
276	                   wtCross and
277	                   wtCrossDown and
278	                   hwtVwap < sommiVwapBearLevel
279	
280	    bullPattern = rsimfi > soomiRSIMFIBullLevel and
281	                   wt2 < soomiFlagWTBullLevel and
282	                   wtCross and
283	                   wtCrossUp and
284	                   hwtVwap > sommiVwapBullLevel
285	
286	    [bearPattern, bullPattern, hwtVwap]
287	
288	f_findSommiDiamond(tf, tf2, wt1, wt2, wtCross, wtCrossUp, wtCrossDown) =>
289	    [candleBodyDir, newBar] = f_getTFCandle(tf)
290	    [candleBodyDir2, newBar2] = f_getTFCandle(tf2)
291	    bearPattern = wt2 >= soomiDiamondWTBearLevel and
292	                   wtCross and
293	                   wtCrossDown and
294	                   not candleBodyDir and
295	                   not candleBodyDir2
296	    bullPattern = wt2 <= soomiDiamondWTBullLevel and
297	                   wtCross and
298	                   wtCrossUp and
299	                   candleBodyDir and
300	                   candleBodyDir2
301	    [bearPattern, bullPattern]
302	
303	// } FUNCTIONS
304	
305	// CALCULATE INDICATORS {
306	
307	// RSI
308	rsi = rsi(rsiSRC, rsiLen)
309	rsiColor = rsi <= rsiOversold ? colorGreen : rsi >= rsiOverbought ? colorRed : colorPurple
310	
311	// RSI + MFI Area
312	rsiMFI = f_rsimfi(rsiMFIperiod, rsiMFIMultiplier, timeframe.period)
313	rsiMFIColor = rsiMFI > 0 ? #3ee145 : #ff3d2e
314	
315	// Calculates WaveTrend
316	[wt1, wt2, wtOversold, wtOverbought, wtCross, wtCrossUp, wtCrossDown, wtCross_last, wtCrossUp_last, wtCrossDown_last, wtVwap] = f_wavetrend(wtMASource, wtChannelLen, wtAverageLen, wtMALen, timeframe.period)
317	
318	// Stochastic RSI
319	[stochK, stochD] = f_stochrsi(stochSRC, stochLen, stochRsiLen, stochKSmooth, stochDSmooth, stochUseLog, stochAvg)
320	
321	// Schaff Trend Cycle
322	tcVal = f_tc(tcSRC, tclength, tcfastLength, tcslowLength)
323	
324	// Sommi flag
325	[sommiBearish, sommiBullish, hvwap] = f_findSommiFlag(sommiVwapTF, wt1, wt2, rsiMFI, wtCross,  wtCrossUp, wtCrossDown)
326	
327	//Sommi diamond
328	[sommiBearishDiamond, sommiBullishDiamond] = f_findSommiDiamond(sommiHTCRes, sommiHTCRes2, wt1, wt2, wtCross, wtCrossUp, wtCrossDown)
329	
330	// macd colors
331	[macdWT1Color, macdWT2Color] = f_macdWTColors(macdWTColorsTF)
332	
333	// WT Divergences
334	[wtFractalTop, wtFractalBot, wtLow_prev, wtBearDiv, wtBullDiv, wtBearDivHidden, wtBullDivHidden] = f_findDivs(wt2, wtDivOBLevel, wtDivOSLevel, true)
335	
336	[wtFractalTop_add, wtFractalBot_add, wtLow_prev_add, wtBearDiv_add, wtBullDiv_add, wtBearDivHidden_add, wtBullDivHidden_add] =  f_findDivs(wt2, wtDivOBLevel_add, wtDivOSLevel_add, true)
337	[wtFractalTop_nl, wtFractalBot_nl, wtLow_prev_nl, wtBearDiv_nl, wtBullDiv_nl, wtBearDivHidden_nl, wtBullDivHidden_nl] =  f_findDivs(wt2, 0, 0, false)
338	
339	wtBearDivHidden_ = showHiddenDiv_nl ? wtBearDivHidden_nl : wtBearDivHidden
340	wtBullDivHidden_ = showHiddenDiv_nl ? wtBullDivHidden_nl : wtBullDivHidden
341	
342	wtBearDivColor = (wtShowDiv and wtBearDiv) or (wtShowHiddenDiv and wtBearDivHidden_) ? colorRed : na
343	wtBullDivColor = (wtShowDiv and wtBullDiv) or (wtShowHiddenDiv and wtBullDivHidden_) ? colorGreen : na
344	
345	wtBearDivColor_add = (wtShowDiv and (wtDivOBLevel_addshow and wtBearDiv_add)) or (wtShowHiddenDiv and (wtDivOBLevel_addshow and wtBearDivHidden_add)) ? #9a0202 : na
346	wtBullDivColor_add = (wtShowDiv and (wtDivOBLevel_addshow and wtBullDiv_add)) or (wtShowHiddenDiv and (wtDivOBLevel_addshow and wtBullDivHidden_add)) ? #1b5e20 : na
347	
348	// RSI Divergences
349	[rsiFractalTop, rsiFractalBot, rsiLow_prev, rsiBearDiv, rsiBullDiv, rsiBearDivHidden, rsiBullDivHidden] = f_findDivs(rsi, rsiDivOBLevel, rsiDivOSLevel, true)
350	[rsiFractalTop_nl, rsiFractalBot_nl, rsiLow_prev_nl, rsiBearDiv_nl, rsiBullDiv_nl, rsiBearDivHidden_nl, rsiBullDivHidden_nl] = f_findDivs(rsi, 0, 0, false)
351	
352	rsiBearDivHidden_ = showHiddenDiv_nl ? rsiBearDivHidden_nl : rsiBearDivHidden
353	rsiBullDivHidden_ = showHiddenDiv_nl ? rsiBullDivHidden_nl : rsiBullDivHidden
354	
355	rsiBearDivColor = (rsiShowDiv and rsiBearDiv) or (rsiShowHiddenDiv and rsiBearDivHidden_) ? colorRed : na
356	rsiBullDivColor = (rsiShowDiv and rsiBullDiv) or (rsiShowHiddenDiv and rsiBullDivHidden_) ? colorGreen : na
357	
358	// Stoch Divergences
359	[stochFractalTop, stochFractalBot, stochLow_prev, stochBearDiv, stochBullDiv, stochBearDivHidden, stochBullDivHidden] = f_findDivs(stochK, 0, 0, false)
360	
361	stochBearDivColor = (stochShowDiv and stochBearDiv) or (stochShowHiddenDiv and stochBearDivHidden) ? colorRed : na
362	stochBullDivColor = (stochShowDiv and stochBullDiv) or (stochShowHiddenDiv and stochBullDivHidden) ? colorGreen : na
363	
364	// Small Circles WT Cross
365	signalColor = wt2 - wt1 > 0 ? color.red : color.lime
366	
367	// Buy signal.
368	buySignal = wtCross and wtCrossUp and wtOversold
369	
370	buySignalDiv = (wtShowDiv and wtBullDiv) or
371	               (wtShowDiv and wtBullDiv_add) or
372	               (stochShowDiv and stochBullDiv) or
373	               (rsiShowDiv and rsiBullDiv)
374	
375	buySignalDiv_color = wtBullDiv ? colorGreen :
376	                     wtBullDiv_add ? color.new(colorGreen, 60) :
377	                     rsiShowDiv ? colorGreen : na
378	
379	// Sell signal
380	sellSignal = wtCross and wtCrossDown and wtOverbought
381	
382	sellSignalDiv = (wtShowDiv and wtBearDiv) or
383	               (wtShowDiv and wtBearDiv_add) or
384	               (stochShowDiv and stochBearDiv) or
385	               (rsiShowDiv and rsiBearDiv)
386	
387	sellSignalDiv_color = wtBearDiv ? colorRed :
388	                     wtBearDiv_add ? color.new(colorRed, 60) :
389	                     rsiBearDiv ? colorRed : na
390	
391	// Gold Buy
392	lastRsi = valuewhen(wtFractalBot, rsi[2], 0)[2]
393	wtGoldBuy = ((wtShowDiv and wtBullDiv) or (rsiShowDiv and rsiBullDiv)) and
394	           wtLow_prev <= osLevel3 and
395	           wt2 > osLevel3 and
396	           wtLow_prev - wt2 <= -5 and
397	           lastRsi < 30
398	
399	// } CALCULATE INDICATORS
400	
401	// DRAW {
402	bgcolor(darkMode ? color.new(#000000, 80) : na)
403	zLine = plot(0, color = color.new(colorWhite, 50))
404	
405	//  MFI BAR
406	rsiMfiBarTopLine = plot(rsiMFIShow ? -95 : na, title = 'MFI Bar TOP Line', transp = 100)
407	rsiMfiBarBottomLine = plot(rsiMFIShow ? -99 : na, title = 'MFI Bar BOTTOM Line', transp = 100)
408	fill(rsiMfiBarTopLine, rsiMfiBarBottomLine, title = 'MFI Bar Colors', color = rsiMFIColor, transp = 75)
409	
410	// WT Areas
411	plot(wtShow ? wt1 : na, style = plot.style_area, title = 'WT Wave 1', color = macdWTColorsShow ? macdWT1Color : colorWT1, transp = 0)
412	plot(wtShow ? wt2 : na, style = plot.style_area, title = 'WT Wave 2', color = macdWTColorsShow ? macdWT2Color : darkMode ? colorWT2_ : colorWT2 , transp = 20)
413	
414	// VWAP
415	plot(vwapShow ? wtVwap : na, title = 'VWAP', color = colorYellow, style = plot.style_area, linewidth = 2, transp = 45)
416	
417	// MFI AREA
418	rsiMFIplot = plot(rsiMFIShow ? rsiMFI: na, title = 'RSI+MFI Area', color = rsiMFIColor, transp = 20)
419	fill(rsiMFIplot, zLine, rsiMFIColor, transp = 40)
420	
421	// WT Div
422	
423	plot(series = wtFractalTop ? wt2[2] : na, title = 'WT Bearish Divergence', color = wtBearDivColor, linewidth = 2, offset = -2)
424	plot(series = wtFractalBot ? wt2[2] : na, title = 'WT Bullish Divergence', color = wtBullDivColor, linewidth = 2, offset = -2)
425	
426	// WT 2nd Div
427	plot(series = wtFractalTop_add ? wt2[2] : na, title = 'WT 2nd Bearish Divergence', color = wtBearDivColor_add, linewidth = 2, offset = -2)
428	plot(series = wtFractalBot_add ? wt2[2] : na, title = 'WT 2nd Bullish Divergence', color = wtBullDivColor_add, linewidth = 2, offset = -2)
429	
430	// RSI
431	plot(rsiShow ? rsi : na, title = 'RSI', color = rsiColor, linewidth = 2, transp = 25)
432	
433	// RSI Div
434	plot(series = rsiFractalTop ? rsi[2] : na, title='RSI Bearish Divergence', color = rsiBearDivColor, linewidth = 1, offset = -2)
435	plot(series = rsiFractalBot ? rsi[2] : na, title='RSI Bullish Divergence', color = rsiBullDivColor, linewidth = 1, offset = -2)
436	
437	// Stochastic RSI
438	stochKplot = plot(stochShow ? stochK : na, title = 'Stoch K', color = color.new(#21baf3, 0), linewidth = 2)
439	stochDplot = plot(stochShow ? stochD : na, title = 'Stoch D', color = color.new(#673ab7, 60), linewidth = 1)
440	stochFillColor = stochK >= stochD ? color.new(#21baf3, 75) : color.new(#673ab7, 60)
441	fill(stochKplot, stochDplot, title='KD Fill', color=stochFillColor)
442	
443	// Stoch Div
444	plot(series = stochFractalTop ? stochK[2] : na, title='Stoch Bearish Divergence', color = stochBearDivColor, linewidth = 1, offset = -2)
445	plot(series = stochFractalBot ? stochK[2] : na, title='Stoch Bullish Divergence', color = stochBullDivColor, linewidth = 1, offset = -2)
446	
447	// Schaff Trend Cycle
448	plot(tcLine ? tcVal : na, color = color.new(#673ab7, 25), linewidth = 2, title = "Schaff Trend Cycle 1")
449	plot(tcLine ? tcVal : na, color = color.new(colorWhite, 50), linewidth = 1, title = "Schaff Trend Cycle 2")
450	
451	// Draw Overbought & Oversold lines
452	//plot(obLevel, title = 'Over Bought Level 1', color = colorWhite, linewidth = 1, style = plot.style_circles, transp = 85)
453	plot(obLevel2, title = 'Over Bought Level 2', color = colorWhite, linewidth = 1, style = plot.style_stepline, transp = 85)
454	plot(obLevel3, title = 'Over Bought Level 3', color = colorWhite, linewidth = 1, style = plot.style_circles, transp = 95)
455	
456	//plot(osLevel, title = 'Over Sold Level 1', color = colorWhite, linewidth = 1, style = plot.style_circles, transp = 85)
457	plot(osLevel2, title = 'Over Sold Level 2', color = colorWhite, linewidth = 1, style = plot.style_stepline, transp = 85)
458	
459	// Sommi flag
460	plotchar(sommiFlagShow and sommiBearish ? 108 : na, title = 'Sommi bearish flag', char='⚑', color = colorPink, location = location.absolute, size = size.tiny, transp = 0)
461	plotchar(sommiFlagShow and sommiBullish ? -108 : na, title = 'Sommi bullish flag', char='⚑', color = colorBluelight, location = location.absolute, size = size.tiny, transp = 0)
462	plot(sommiShowVwap ? ema(hvwap, 3) : na, title = 'Sommi higher VWAP', color = colorYellow, linewidth = 2, style = plot.style_line, transp = 15)
463	
464	// Sommi diamond
465	plotchar(sommiDiamondShow and sommiBearishDiamond ? 108 : na, title = 'Sommi bearish diamond', char='◆', color = colorPink, location = location.absolute, size = size.tiny, transp = 0)
466	plotchar(sommiDiamondShow and sommiBullishDiamond ? -108 : na, title = 'Sommi bullish diamond', char='◆', color = colorBluelight, location = location.absolute, size = size.tiny, transp = 0)
467	
468	// Circles
469	plot(wtCross ? wt2 : na, title = 'Buy and sell circle', color = signalColor, style = plot.style_circles, linewidth = 3, transp = 15)
470	
471	plotchar(wtBuyShow and buySignal ? -107 : na, title = 'Buy circle', char='·', color = colorGreen, location = location.absolute, size = size.small, transp = 50)
472	plotchar(wtSellShow and sellSignal ? 105 : na , title = 'Sell circle', char='·', color = colorRed, location = location.absolute, size = size.small, transp = 50)
473	
474	plotchar(wtDivShow and buySignalDiv ? -106 : na, title = 'Divergence buy circle', char='•', color = buySignalDiv_color, location = location.absolute, size = size.small, offset = -2, transp = 15)
475	plotchar(wtDivShow and sellSignalDiv ? 106 : na, title = 'Divergence sell circle', char='•', color = sellSignalDiv_color, location = location.absolute, size = size.small, offset = -2, transp = 15)
476	
477	plotchar(wtGoldBuy and wtGoldShow ? -106 : na, title = 'Gold  buy gold circle', char='•', color = colorOrange, location = location.absolute, size = size.small, offset = -2, transp = 15)
478	
479	// } DRAW
480	
481	// ALERTS {
482	
483	// BUY
484	alertcondition(buySignal, 'Buy (Big green circle)', 'Green circle WaveTrend Oversold')
485	alertcondition(buySignalDiv, 'Buy (Big green circle + Div)', 'Buy & WT Bullish Divergence & WT Overbought')
486	alertcondition(wtGoldBuy, 'GOLD Buy (Big GOLDEN circle)', 'Green & GOLD circle WaveTrend Overbought')
487	alertcondition(sommiBullish or sommiBullishDiamond, 'Sommi bullish flag/diamond', 'Blue flag/diamond')
488	alertcondition(wtCross and wtCrossUp, 'Buy (Small green dot)', 'Buy small circle')
489	
490	// SELL
491	alertcondition(sommiBearish or sommiBearishDiamond, 'Sommi bearish flag/diamond', 'Purple flag/diamond')
492	alertcondition(sellSignal, 'Sell (Big red circle)', 'Red Circle WaveTrend Overbought')
493	alertcondition(sellSignalDiv, 'Sell (Big red circle + Div)', 'Buy & WT Bearish Divergence & WT Overbought')
494	alertcondition(wtCross and wtCrossDown, 'Sell (Small red dot)', 'Sell small circle')
495	
496	// } ALERTS
```

---

*These Pine Script sources are provided as reference for the Python implementation. The AI Trading Bot will re-implement these algorithms using pandas, numpy, and pandas-ta for full local control and integration with the LangChain decision engine.*