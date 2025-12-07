

---



\### `docs/MASTER\_SPEC.md` (new)



```markdown

\# MASTER SPEC –

---



\### `docs/MASTER\_SPEC.md` (new)



```markdown

\# MASTER SPEC – muni-lab-core (v0.1)



> Living specification for the municipal bond analytics core.

> Code and spec are meant to evolve together.



---



\## 1. Goals and Scope



\- Provide a \*\*clean, reusable Python engine\*\* for municipal bond analytics.

\- Keep \*\*personal data and broker-specific plumbing out\*\* of this repo.

\- Focus on:

&nbsp; - Yield curves and forward curves

&nbsp; - Z-spread and OAS-based pricing

&nbsp; - Callable bond logic and call likelihood diagnostics

&nbsp; - NPV comparisons (old vs new bond, call vs no-call, horizon vs maturity)

&nbsp; - Risk metrics: KRD / KRC, simple scenario tools



This spec documents the intended behavior of each module under `src/muni\_core`.



---



\## 2. Core Data Model (Bond Row)



\*\*Goal:\*\* define the minimal set of fields required to run pricing, call, and NPV logic.



Draft fields (to refine later):



\- `CUSIP`: string

\- `Rating`: string (e.g. "AA1")

\- `RatingNum`: numeric mapping for rating buckets

\- `Basis`: day count / basis convention (e.g. "Actual/Actual")

\- `SettleDate`: trade / valuation date

\- `Coupon`: annual coupon rate (percent)

\- `MaturityDate`

\- `CallDate` (first call date, if any)

\- `CallPrice` (typically 100)

\- `Quantity`

\- `MarketPrice\_Clean`

\- `Z\_spread\_bp`

\- `OAS\_bp`

\- `CurveBucket` or `CurveRating` (e.g. AAA / AA / A1 etc.)



Later, this will sync with your master Excel control sheet.



---



\## 3. Curves Module (`muni\_core.curves`)



Responsibilities:



\- Load \*\*AAA wide curves\*\* (tenor vs yield by rating bucket).

\- Load \*\*AAA spot curve\*\* for discounting and forward-rate generation.

\- Provide helper functions:

&nbsp; - `build\_zero\_curve(...)`

&nbsp; - `discount\_factors(t)`

&nbsp; - `forward\_rate(start, end)`

&nbsp; - `get\_bucket\_curve(rating\_num)` (map to AAA/AA/A1 etc.)



HW / short-rate model hooks live here as well, but can be layered later.



---



\## 4. Call / Option Module (`muni\_core.calls`)



Responsibilities:



\- Represent call features (first call, Bermudan window, 6-month intervals, etc.).

\- Implement a \*\*simple NPV call test\*\* at “best call date”.

\- Provide labels / diagnostics such as:

&nbsp; - `Call\_Likelihood\_Label`

&nbsp; - `Call\_Likelihood\_Score`

&nbsp; - `BestCallDate`

&nbsp; - `BestCallForward`

&nbsp; - `BestCallGap\_bp`



This module will wrap:

\- Your current “NPV savings > 3% ⇒ high call likelihood” rule.

\- Additional logic for \*\*pre-call NPV profiles\*\*.



---



\## 5. NPV / Cashflow Module (`muni\_core.npv`)



Responsibilities:



\- Generate cashflows for:

&nbsp; - \*\*No-call path\*\* (to maturity).

&nbsp; - \*\*Call path\*\* (to a specific call date).

\- Discount using the chosen curve + spread:

&nbsp; - Price at Z-spread

&nbsp; - Price at OAS

\- Support:

&nbsp; - Horizon analysis (e.g. 10-year horizon but pull PV back from maturity).

&nbsp; - EV-style outputs such as `EV\_PV\_NoCall\_Principal`.



This is where we fix the “truncate at horizon vs follow to maturity then PV-back” issue.



---



\## 6. Risk Module (`muni\_core.krd\_krc`)



Responsibilities:



\- Compute KRD / KRC for:

&nbsp; - Parallel shocks

&nbsp; - Bucketed key-rate shocks

\- Initially: use the same curve used for OAS pricing.

\- Later: optionally allow separate “risk curve” if needed.



---



\## 7. Configuration Module (`muni\_core.config`)



Responsibilities:



\- Read a central config (YAML / Excel / INI).

\- Provide a clean `Config` object to the rest of the code:

&nbsp; - File paths (curve files, controls workbook).

&nbsp; - Curve strategy (`excel\_curves\_wide`, etc.).

&nbsp; - Sigma mode, sigma grid.

&nbsp; - Call / NPV thresholds (e.g. `NPV\_call\_threshold = 0.03`).



This module is the bridge to your \*\*MUNI\_MASTER\_BUCKET\*\* workbook.



---



\## 8. Roadmap



Near-term:



1\. Lock the bond-row schema and config pattern.

2\. Implement a simple curve loader + discount factors.

3\. Port the existing `muni\_oas\_z\_forward\_call\_v4\_swap.py` into:

&nbsp;  - `curves`

&nbsp;  - `calls`

&nbsp;  - `npv`

4\. Add smoke tests for a handful of bonds.



Later:



\- Default-risk module.

\- Machine learning hooks for call probability.

\- CLI / Streamlit front-end.

&nbsp;muni-lab-core (v0.1)



> Living specification for the municipal bond analytics core.

> Code and spec are meant to evolve together.



---



\## 1. Goals and Scope



\- Provide a \*\*clean, reusable Python engine\*\* for municipal bond analytics.

\- Keep \*\*personal data and broker-specific plumbing out\*\* of this repo.

\- Focus on:

&nbsp; - Yield curves and forward curves

&nbsp; - Z-spread and OAS-based pricing

&nbsp; - Callable bond logic and call likelihood diagnostics

&nbsp; - NPV comparisons (old vs new bond, call vs no-call, horizon vs maturity)

&nbsp; - Risk metrics: KRD / KRC, simple scenario tools



This spec documents the intended behavior of each module under `src/muni\_core`.



---



\## 2. Core Data Model (Bond Row)



\*\*Goal:\*\* define the minimal set of fields required to run pricing, call, and NPV logic.



Draft fields (to refine later):



\- `CUSIP`: string

\- `Rating`: string (e.g. "AA1")

\- `RatingNum`: numeric mapping for rating buckets

\- `Basis`: day count / basis convention (e.g. "Actual/Actual")

\- `SettleDate`: trade / valuation date

\- `Coupon`: annual coupon rate (percent)

\- `MaturityDate`

\- `CallDate` (first call date, if any)

\- `CallPrice` (typically 100)

\- `Quantity`

\- `MarketPrice\_Clean`

\- `Z\_spread\_bp`

\- `OAS\_bp`

\- `CurveBucket` or `CurveRating` (e.g. AAA / AA / A1 etc.)



Later, this will sync with your master Excel control sheet.



---



\## 3. Curves Module (`muni\_core.curves`)



Responsibilities:



\- Load \*\*AAA wide curves\*\* (tenor vs yield by rating bucket).

\- Load \*\*AAA spot curve\*\* for discounting and forward-rate generation.

\- Provide helper functions:

&nbsp; - `build\_zero\_curve(...)`

&nbsp; - `discount\_factors(t)`

&nbsp; - `forward\_rate(start, end)`

&nbsp; - `get\_bucket\_curve(rating\_num)` (map to AAA/AA/A1 etc.)



HW / short-rate model hooks live here as well, but can be layered later.



---



\## 4. Call / Option Module (`muni\_core.calls`)



Responsibilities:



\- Represent call features (first call, Bermudan window, 6-month intervals, etc.).

\- Implement a \*\*simple NPV call test\*\* at “best call date”.

\- Provide labels / diagnostics such as:

&nbsp; - `Call\_Likelihood\_Label`

&nbsp; - `Call\_Likelihood\_Score`

&nbsp; - `BestCallDate`

&nbsp; - `BestCallForward`

&nbsp; - `BestCallGap\_bp`



This module will wrap:

\- Your current “NPV savings > 3% ⇒ high call likelihood” rule.

\- Additional logic for \*\*pre-call NPV profiles\*\*.



---



\## 5. NPV / Cashflow Module (`muni\_core.npv`)



Responsibilities:



\- Generate cashflows for:

&nbsp; - \*\*No-call path\*\* (to maturity).

&nbsp; - \*\*Call path\*\* (to a specific call date).

\- Discount using the chosen curve + spread:

&nbsp; - Price at Z-spread

&nbsp; - Price at OAS

\- Support:

&nbsp; - Horizon analysis (e.g. 10-year horizon but pull PV back from maturity).

&nbsp; - EV-style outputs such as `EV\_PV\_NoCall\_Principal`.



This is where we fix the “truncate at horizon vs follow to maturity then PV-back” issue.



---



\## 6. Risk Module (`muni\_core.krd\_krc`)



Responsibilities:



\- Compute KRD / KRC for:

&nbsp; - Parallel shocks

&nbsp; - Bucketed key-rate shocks

\- Initially: use the same curve used for OAS pricing.

\- Later: optionally allow separate “risk curve” if needed.



---



\## 7. Configuration Module (`muni\_core.config`)



Responsibilities:



\- Read a central config (YAML / Excel / INI).

\- Provide a clean `Config` object to the rest of the code:

&nbsp; - File paths (curve files, controls workbook).

&nbsp; - Curve strategy (`excel\_curves\_wide`, etc.).

&nbsp; - Sigma mode, sigma grid.

&nbsp; - Call / NPV thresholds (e.g. `NPV\_call\_threshold = 0.03`).



This module is the bridge to your \*\*MUNI\_MASTER\_BUCKET\*\* workbook.



---



\## 8. Roadmap



Near-term:



1\. Lock the bond-row schema and config pattern.

2\. Implement a simple curve loader + discount factors.

3\. Port the existing `muni\_oas\_z\_forward\_call\_v4\_swap.py` into:

&nbsp;  - `curves`

&nbsp;  - `calls`

&nbsp;  - `npv`

4\. Add smoke tests for a handful of bonds.



Later:



\- Default-risk module.

\- Machine learning hooks for call probability.

\- CLI / Streamlit front-end.



