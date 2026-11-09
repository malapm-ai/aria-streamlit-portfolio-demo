# ARIA Product Specification — Portfolio Release

## Product goal

ARIA is an adaptive banking resolution agent for a portfolio project. It should feel like a capable customer-support specialist: understand the customer's issue, select a relevant investigation, use evidence and applicable policy, adapt when the findings change, and explain the outcome in plain English.

This release is an open portfolio demo using fictional records only. It must be impossible to connect this release to production customer data or execute financial actions.

## Three product views

### Customer

- Public entry to a clearly labelled synthetic-data demo.
- Customer ID pattern is a demo lookup, not identity authentication.
- Show a fictional read-only profile with sensitive fields masked.
- Intake includes issue category, free-text description, optional amount/reference/date, file evidence, and English voice-to-text when supported.
- Create a unique ticket and preserve it across the investigation.
- Use a case-specific plan. Select only relevant checks; skip irrelevant tools; reconsider the plan when evidence conflicts or a customer answers a clarification.
- Ask a focused multiple-choice or free-text clarification when a missing fact materially affects the result. Let the customer say they do not know or cancel.
- Present concise conversational timeline events: what ARIA understood, what it checked and why, verified findings, uncertainty, and next step.
- Finish with one of: evidence-backed information, customer guidance, a focused information request, or a staff-review recommendation.
- Ask whether the customer is satisfied. An unsatisfied customer can request staff review using the same ticket.

### Customer support

- Show a synthetic queue ordered by risk/urgency, age, and customer impact.
- Let a demo support user claim or assign a case and record when it was opened/acknowledged.
- Show a one-page case brief: customer's request, checks performed, source-backed findings, evidence verification status, applicable policy references, uncertainty/conflicts, risk, and ARIA's recommendation.
- Make evidence easy to verify without requiring a long PDF. Any downloadable handoff should be concise and preserve source references.
- Let support record a decision, request more information, or return a case to ARIA. These actions are simulated in the portfolio release.
- Customer-visible assignment and "seen" status must come from saved case events; never display a fictional staff member as having viewed a case unless that event exists.

### Management

- Aggregate portfolio-demo metrics for volume, issue mix, completion, escalation, repeat contact, customer satisfaction, time to first response/resolution, tool reliability, latency, clarification count, loop stops, and estimated model/tool cost.
- Identify all demo-only, simulated, or estimated measures in the UI. Do not present estimated savings or revenue as actual bank performance.
- Support analysis of where ARIA needs better policy coverage, tools, UX, or language quality.
- Default to aggregate data; individual case details require a support role.

## Agent behavior

- The investigation is adaptive, not a fixed checklist. ARIA can choose no tools, one tool, or several, based on the issue and current evidence.
- Keep deterministic policy/safety gates around an adaptive planner. The planner may propose checks; only authorized tools and validated rules may run.
- Re-plan after new evidence or a customer answer. Stop on a verified answer, an actionable guidance outcome, a required clarification, a risk boundary, an unavailable dependency, or an exhausted budget.
- Do not expose raw hidden chain-of-thought. Show concise customer-facing rationale and actual check results. Keep detailed tool traces and audit data in the support view.
- Distinguish uploaded customer evidence from bank-verified records. A file's appearance alone does not establish authenticity.
- Cite only policy references returned by the policy source. If a relevant policy is missing or conflicting, explain the uncertainty and route to staff.
- Never claim a refund, reversal, card block, merchant contact, staff notification, or case status change unless the authoritative system confirms it.

## Safety and data boundaries

- All 40,000 customer profiles and transactions used in this public release are fictional demo records.
- The public demo must use a separate synthetic Supabase project or a demonstrably isolated synthetic dataset. Never point public configuration at a production database or production service-role key.
- Do not mix fictional portfolio records with real customer tables.
- Public customer-ID lookup is permitted only for synthetic records and must be labelled as a demo lookup, not secure sign-in.
- No real email/push notifications are sent in v1. Support assignment and seen-state are in-app demo events.
- No money movement or consequential banking action is performed by ARIA. All financial actions require a human decision, and remain simulated in this release.
- Management cost/revenue figures are estimates or synthetic analytics, not claims about actual savings.
- Public data writes are limited to demo cases, feedback, and demo workflow events; customer profiles, balances, and transactions are read-only.

## Technology story for the portfolio

- Streamlit: three role-based product views.
- Existing FastAPI ARIA service: source of agent behavior and API contracts, subject to public-demo safety review.
- Supabase: synthetic customer/case data only for this release.
- Pinecone/RAG: policy and operational guidance retrieval, with source references.
- Mistral/OpenRouter: configured model providers; do not claim current provider availability without a live check.
- LangGraph: planner-selection shadow slice in the current candidate, not the full investigation loop. Do not describe the entire agent as LangGraph-driven.
- Management architecture panel should distinguish implemented, simulated, and planned capabilities.
- Explain the adaptive loop using verified source behavior: plan from the issue and evidence, run an allowed check, observe its result, and re-plan or stop under explicit iteration/provider budgets. Show only deployed configuration values; never present documentation defaults as live configuration.
- Candidate-source defaults observed for explanation include planner temperature 0.0, maximum 9 agent iterations, planner-call budget 4 (bounded at 5), and maximum 2 consecutive tool failures (bounded at 3). Verify the actual deployed settings before displaying them.

## Existing implementation caveat

- The root `app.py` is an older Streamlit prototype. It generates mock customer profiles, mock voice transcription, mock tool-thinking text, mock support staff, and mock PDF output. Its `save_to_supabase_silent()` function is a no-op. It is not the active Supabase-backed ARIA agent and must not be presented as live behavior.
- The adjacent older `phase3_llm_integration.py` contains a ₹10,000 auto-approval rule. That behavior conflicts with the current product requirement and must not be reused. The portfolio release simulates financial recommendations only; a human approval is required for every money action.
- The current agent candidate is the consolidated FastAPI implementation. The 9.10.7 file adds a LangGraph planner-selection shadow comparison; it does not replace the full investigation loop.

## Future live-customer product — separate release

- Separate deployment and data configuration from the public portfolio demo.
- Customer identity must be verified before any profile, account, or transaction data is shown. A predictable customer ID is not authentication.
- Staff and management use server-verified staff roles; customers can access only their own records.
- Keep service-role credentials server-side and enforce row-level authorization in the backend/database.
- Human approval remains mandatory for every money action.
- Add actual staff notification integrations only after provider, authorization, recipient, delivery, and audit behavior are defined.
- Complete security, privacy, policy, and live-data review before exposing real customer records.

## Release scope decision

The user wants an open, recruiter-testable portfolio website. The build target is therefore the synthetic-data portfolio release. A future private live-customer service is a separate product deployment, not a mode switch on the public demo.
