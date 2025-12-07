# muni-lab-core

Municipal Bond Analytics



\# muni-lab-core



Municipal Bond Analytics – core engine.



This repo is the \*\*framework layer\*\* for a larger muni bond project:

\- Yield curves and forward curves

\- Z-spread and OAS calculations

\- Callable bond / call-probability logic

\- NPV / swap profile and horizon analysis

\- Key-rate duration (KRD) and key-rate convexity (KRC)



Personal data (CUSIPs, broker exports, live positions) should \*\*not\*\* live here.

Those stay in private spreadsheets / repos and connect via CSV or config files.



\## Layout



```text

src/muni\_core/

&nbsp; config/   # configuration loading and validation

&nbsp; curves/   # AAA muni curves, spot curves, forward rates, HW parameters

&nbsp; calls/    # call window logic, NPV call test, Bermudan-style helpers

&nbsp; npv/      # cashflow and NPV engines (call / no-call / horizon)

&nbsp; krd\_krc/  # risk metrics on top of the pricing engines

docs/

&nbsp; MASTER\_SPEC.md  # living specification for the models and modules



