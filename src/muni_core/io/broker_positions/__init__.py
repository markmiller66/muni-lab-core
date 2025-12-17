"""
muni_core.io.broker_positions

Broker position ingestion + MSRB enrichment pipeline.

Stages:
1) build_positions.py  -> standardized combined positions file
2) enrich_with_msrb.py -> fills coupon/call/maturity/ratings from MSRB

Keep orchestration in stage scripts; keep parsing in loaders; keep helpers in utils.
"""
