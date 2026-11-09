# ARIA · Streamlit Community Cloud deployment guide

This guide deploys the public portfolio build in `streamlit_app.py`. It is designed for fictional demo records only. Do not connect a production Supabase project or production credentials.

## Before publishing

1. Confirm the configured Supabase project contains only synthetic demo customers, accounts, transactions, cases, and feedback.
2. Use a dedicated Supabase project for the public demo. Keep its server-side key only in Streamlit Cloud secrets. The app checks that the URL project reference matches `ARIA_SYNTHETIC_SUPABASE_PROJECT_REF`.
3. Ensure the synthetic database has the tables and policy/RAG data expected by the existing ARIA backend. The app does not create or seed the 40,000 records.
4. Add abuse controls before advertising the public URL. This app has no customer login, and predictable demo IDs are intentionally not authentication. A public visitor can consume model/database resources. Put request limits and spend alerts at the hosting/provider layer and keep the demo dataset isolated.
5. Do not commit `.env`, `streamlit_secrets.toml`, `.streamlit/secrets.toml`, API keys, service-role keys, or customer exports.

## Put the app in GitHub

Create a GitHub repository containing the app entry point, the ARIA backend module, `aria_langgraph_bridge.py`, `requirements.txt`, and `.streamlit/config.toml`. Do not upload the full historical handoff pack or large CSV exports. Keep `streamlit_app.py` and the backend module in the same directory.

## Deploy

1. Connect the GitHub account to Streamlit Community Cloud.
2. Choose **Create app**, select the repository and branch, and set the entrypoint to `streamlit_app.py`.
3. In **Advanced settings**, select Python 3.11 for the current legacy dependency pins in `requirements.txt`.
4. In **Secrets**, copy the template from `.streamlit/secrets.toml.example`, then replace the placeholders with values for the synthetic-only Supabase project and whichever model/RAG providers are actually configured.
5. Deploy and review the app logs. The page should refuse to start if the data mode or pinned Supabase project reference is missing or mismatched.
6. Before sharing the URL, manually confirm profile lookup, ticket creation, agent response, support queue, and feedback against synthetic records. No validation or Cloud deployment has been run from this workspace.

Streamlit Community Cloud deploys from a GitHub repository; its deployment flow lets you select the entrypoint, Python version, and secrets. See [Streamlit deployment docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Current feature status

- Included in the app scaffold: synthetic profile lookup, KYC gating, masked profile fields, issue intake, optional image/PDF evidence, ticket investigation, customer-safe timeline, clarification response, feedback, in-app support view, and synthetic management metrics.
- Not included yet: browser voice dictation, durable public customer authentication, verified staff accounts, production notifications, model-token cost telemetry, or a publish-ready rate limiter. No real money movement is available.
- Policy citations are shown only when the agent event contains citations. Uploaded files remain customer-provided evidence unless backend checks independently corroborate them.
- Streamlit's public app and the legacy customer-ID path are not suitable for real customer information. A future live-data deployment must be separate and add customer authentication and server-side authorization.

