"""Public ARIA portfolio app. Synthetic data only; never a live-customer app."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import streamlit as st

st.set_page_config(
    page_title="ARIA | Banking Resolution Agent",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "") or "").strip()
    except Exception:
        return ""


def validate_demo_configuration() -> tuple[bool, str]:
    """Refuse to start unless the operator explicitly pins a synthetic Supabase project."""
    if secret("ARIA_DATA_MODE").lower() != "synthetic_demo":
        return False, "Set ARIA_DATA_MODE to synthetic_demo."
    expected_ref = secret("ARIA_SYNTHETIC_SUPABASE_PROJECT_REF")
    supabase_url = secret("SUPABASE_URL")
    # Supabase's current Connect panel calls the server credential
    # SUPABASE_SECRET_KEY. Keep the legacy variable as a fallback for existing
    # deployments; the backend itself currently reads SUPABASE_KEY.
    supabase_key = secret("SUPABASE_SECRET_KEY") or secret("SUPABASE_KEY")
    if not expected_ref or not supabase_url or not supabase_key:
        return False, "Set the synthetic Supabase project reference, URL, and server-side key in Streamlit secrets."
    host = (urlparse(supabase_url).hostname or "").lower()
    if host.split(".")[0] != expected_ref.lower():
        return False, "The configured Supabase URL does not match the pinned synthetic project."

    # Keep secrets server-side. This app is permitted to connect only to the pinned
    # fictional-data project. No production connection string belongs in this app.
    for key in (
        "SUPABASE_URL", "SUPABASE_ANON_KEY",
        "PINECONE_API_KEY", "PINECONE_INDEX_NAME",
        "MISTRAL_API_KEY", "MISTRAL_MODEL",
        "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
        "OPENROUTER_SITE_URL", "ARIA_OPENROUTER_APP_NAME",
        "ARIA_PLANNER_CALL_BUDGET", "MAX_AGENT_ITERATIONS", "MAX_TOOL_FAILURES",
    ):
        value = secret(key)
        if value:
            os.environ[key] = value
    # The private Supabase key is server-side only. The consolidated legacy
    # backend reads SUPABASE_KEY, while current Supabase docs call it SECRET_KEY.
    os.environ["SUPABASE_KEY"] = supabase_key
    os.environ["ARIA_CUSTOMER_ID_ONLY_DEMO_MODE"] = "false"
    os.environ["ARIA_LANGGRAPH_PLANNER_MODE"] = "shadow"
    return True, ""


ready, config_error = validate_demo_configuration()
if not ready:
    st.title("ARIA · Banking Resolution Agent")
    st.warning("The public portfolio demo is not connected.")
    st.write(config_error)
    st.caption("This app accepts fictional records only. Do not configure real customer data here.")
    st.stop()

try:
    import aria_backend_customer_consolidated_v9_10_7_langgraph_shadow as aria
except Exception:
    st.title("ARIA · Banking Resolution Agent")
    st.error("The ARIA agent could not be loaded. Check the deployment configuration.")
    st.stop()

st.markdown(
    """
    <style>
    .block-container{max-width:1440px;padding-top:1.2rem}
    .aria-header{background:#071c2b;color:#f5f8fc;padding:1.1rem 1.4rem;border-radius:16px}
    .aria-sub{color:#c8d7e6;font-size:.95rem}
    .demo-strip{background:#fff7dd;border:1px solid #eed99b;color:#624b0b;padding:.65rem .9rem;border-radius:10px;margin:.8rem 0 1.1rem}
    </style>
    <div class="aria-header"><div style="font-size:1.55rem;font-weight:750">🏦 ARIA · Banking Resolution Agent</div>
    <div class="aria-sub">Adaptive Resolution &amp; Investigation Agent · Secure · Transparent · Customer-first</div></div>
    <div class="demo-strip"><b>Portfolio simulation:</b> all customer and transaction records are fictional. Staff assignment and management analytics are demo workflows.</div>
    """,
    unsafe_allow_html=True,
)


def masked(value: object, keep: int = 4) -> str:
    raw = re.sub(r"\s+", "", str(value or ""))
    if not raw:
        return "Not available"
    return ("•" * max(0, len(raw) - keep)) + raw[-keep:]


def parse_event(raw: str) -> dict:
    for line in str(raw).splitlines():
        if line.startswith("data:"):
            try:
                result = json.loads(line[5:].strip())
                if isinstance(result, dict):
                    return result
            except Exception:
                pass
    return {"event": "AGENT_STATUS", "message": "ARIA recorded an update."}


async def collect_agent_events(context: dict) -> list[dict]:
    events = []
    async for raw in aria.run_agent(context):
        events.append(parse_event(raw))
    return events


async def collect_response_events(response) -> list[dict]:
    events = []
    iterator = getattr(response, "body_iterator", None)
    if iterator is None:
        return events
    async for chunk in iterator:
        raw = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
        for block in raw.split("\n\n"):
            event = parse_event(block)
            if event.get("event"):
                events.append(event)
    return events


def customer_view() -> None:
    st.subheader("Customer")
    st.write("Explore ARIA with a fictional demo Customer ID. This is not bank sign-in.")
    with st.form("profile_lookup"):
        customer_id = st.text_input("Demo Customer ID", placeholder="For example: C000002", max_chars=32)
        lookup = st.form_submit_button("Find demo profile", type="primary")
    if lookup:
        cid = customer_id.strip().upper()
        if not re.fullmatch(r"C\d+", cid):
            st.error("Use a demo ID beginning with C followed by digits.")
            st.session_state.pop("aria_profile", None)
        else:
            with st.spinner("Checking the fictional demo profile…"):
                profile, error = aria.load_customer_portal_profile(cid)
            if error or not profile:
                st.error(error or "The demo profile could not be loaded.")
                st.session_state.pop("aria_profile", None)
            else:
                st.session_state.aria_profile = profile
                st.session_state.aria_customer_id = cid
                st.session_state.aria_events = []

    profile = st.session_state.get("aria_profile")
    if not profile:
        st.info("Your investigation will start after you choose a demo profile.")
        return
    if not profile.get("kyc_verified"):
        st.warning("This fictional profile is not KYC verified. Account details remain hidden for this scenario.")
        return

    customer_id = st.session_state.aria_customer_id
    c1, c2, c3 = st.columns(3)
    c1.metric("Customer", str(profile.get("customer_name") or profile.get("full_name") or profile.get("name") or "Demo customer"))
    c2.metric("Mobile", masked(profile.get("mobile_number") or profile.get("phone_number") or profile.get("phone")))
    c3.metric("Account", masked(profile.get("account_number") or profile.get("account_no")))
    with st.expander("More demo profile details"):
        a, b, c = st.columns(3)
        a.write(f"**IFSC:** {profile.get('ifsc_code') or 'Not available'}")
        b.write(f"**Balance:** {profile.get('currency') or 'INR'} {profile.get('balance') if profile.get('balance') is not None else 'Not available'}")
        c.write(f"**KYC:** {profile.get('kyc_status') or 'Not available'}")

    st.divider()
    st.markdown("#### Tell ARIA what happened")
    with st.form("case_intake"):
        category = st.selectbox("Issue type", [
            "UPI Issue", "Duplicate Transaction", "Refund", "ATM Issue", "Card Issue",
            "Unauthorized Transaction", "Loan Issue", "Account Issue", "KYC / Profile",
            "Billing Error", "Other",
        ])
        description = st.text_area("Describe the problem in your own words", height=115, max_chars=5000)
        x, y, z = st.columns(3)
        amount_text = x.text_input("Amount (optional, INR)", placeholder="e.g. 1,250")
        reference = y.text_input("Transaction reference (optional)", max_chars=200)
        incident_date = z.text_input("Transaction date (optional)", placeholder="YYYY-MM-DD")
        upload = st.file_uploader("Evidence (optional)", type=["png", "jpg", "jpeg", "pdf"])
        submit = st.form_submit_button("Create ticket and investigate", type="primary", disabled=not bool(description.strip()))

    if submit:
        amount = None
        if amount_text.strip():
            try:
                amount = float(amount_text.replace(",", "").strip())
                if amount <= 0:
                    raise ValueError
            except ValueError:
                st.error("Enter a positive amount, or leave it blank.")
                return
        evidence_data, evidence_name = None, None
        if upload is not None:
            content = upload.getvalue()
            if len(content) > 5 * 1024 * 1024:
                st.error("Evidence must be 5 MB or smaller.")
                return
            evidence_data = f"data:{upload.type};base64," + base64.b64encode(content).decode("ascii")
            evidence_name = upload.name
        request = aria.CaseRequest(
            customer_id=customer_id,
            category=category,
            amount=amount,
            transaction_reference=reference.strip() or None,
            dispute_date=incident_date.strip() or None,
            description=description.strip(),
            evidence_file=evidence_data,
            evidence_file_name=evidence_name,
        )
        with st.spinner("ARIA is selecting the checks this case needs…"):
            try:
                context = aria.build_case_context(request)
                events = asyncio.run(collect_agent_events(context))
            except Exception:
                st.error("ARIA could not complete this demo investigation. No financial action was taken.")
                return
        st.session_state.aria_events = events
        st.session_state.aria_case_id = context.get("case_id")

    events = st.session_state.get("aria_events", [])
    if events:
        st.divider()
        st.markdown("#### Your investigation")
        ticket_id = events[0].get("ticket_id") or "Ticket pending"
        st.info(f"Ticket **{ticket_id}** · Updates below come from ARIA's investigation events.")
        for event in events:
            kind = str(event.get("event") or "AGENT_STATUS")
            # Only display customer-safe fields. Never fall back to planner/tool
            # reasoning or raw result payloads.
            if kind == "FINAL_DECISION":
                text = str(event.get("summary") or "ARIA prepared a recommendation from the available evidence.")
            elif kind == "AGENT_QUESTION":
                text = str(event.get("question") or "ARIA needs one more detail to continue.")
            elif kind == "CASE_START":
                text = "Your ticket was created and ARIA is starting the investigation."
            elif kind == "STAGE_START":
                text = str(event.get("customer_message") or event.get("title") or "ARIA is checking relevant information.")
            elif kind in {"TOOL_START", "TOOL_RESULT"}:
                text = str(event.get("customer_message") or event.get("customer_summary") or "ARIA is checking a relevant record.")
            elif kind == "COMPLETE":
                text = "The investigation run has ended. Review the recommendation above and the current case status below."
            elif kind == "ERROR":
                text = "ARIA could not complete this check. The case may need another attempt or staff review."
            else:
                text = str(event.get("customer_message") or event.get("customer_summary") or "ARIA is reviewing the available evidence.")
            if kind == "TOOL_START":
                with st.expander(str(event.get("customer_title") or event.get("tool_label") or "Checking a relevant record")):
                    st.write(text)
                    if event.get("why"):
                        st.caption(str(event["why"]))
            elif kind == "AGENT_QUESTION":
                st.warning(text)
            elif kind in {"COMPLETE", "FINAL_DECISION"}:
                st.success(text)
                if kind == "FINAL_DECISION":
                    citations = event.get("policy_citations") or []
                    if citations:
                        with st.expander("Policy sources ARIA used"):
                            for citation in citations[:8]:
                                if isinstance(citation, dict):
                                    title = citation.get("title") or citation.get("source_name") or citation.get("policy_id") or "Policy record"
                                    reference = citation.get("reference") or citation.get("section") or citation.get("source_reference")
                                    st.write(f"**{title}**" + (f" · {reference}" if reference else ""))
                                    summary = citation.get("summary") or citation.get("excerpt")
                                    if summary:
                                        st.caption(str(summary)[:500])
                                else:
                                    st.write(str(citation)[:500])
            elif kind == "ERROR":
                st.error(text)
            else:
                st.write(f"**{event.get('timeline_stage_label') or kind.replace('_', ' ').title()}**")
                st.write(text)
        st.caption("ARIA provides guidance only. This demo does not move money or contact a real bank employee or merchant.")

        case_id = st.session_state.get("aria_case_id")
        if case_id:
            pending = next((event for event in reversed(events) if event.get("event") == "AGENT_QUESTION"), None)
            if pending:
                st.markdown("#### Help ARIA continue")
                with st.form(f"clarification_{case_id}"):
                    options = [str(x) for x in (pending.get("options") or [])]
                    if options:
                        answer = st.radio(str(pending.get("question") or "Choose the closest answer"), options + ["I don't know / I don't have this information"], key=f"clarification_choice_{case_id}")
                    else:
                        answer = st.text_area(str(pending.get("question") or "Please add the missing detail"), max_chars=4000, key=f"clarification_text_{case_id}")
                    continue_run = st.form_submit_button("Continue investigation", type="primary")
                if continue_run:
                    response_text = str(answer or "").strip() or "I don't know / I don't have this information."
                    identity = {"role": "customer", "customer_id": customer_id, "session_token": "synthetic-demo-session"}
                    with st.spinner("ARIA is updating its investigation with your answer…"):
                        try:
                            response = asyncio.run(
                                aria.respond_to_case(
                                    case_id,
                                    aria.CaseResponseRequest(response_text=response_text),
                                    identity,
                                )
                            )
                            st.session_state.aria_events = asyncio.run(collect_response_events(response))
                            st.rerun()
                        except Exception:
                            st.error("ARIA could not resume this demo case. The existing ticket remains available for staff review.")

            tracking = aria.get_support_tracking(case_id)
            status = str(tracking.get("status") or tracking.get("case_status") or "UNDER_INVESTIGATION")
            st.write(f"**Current case status:** {status}")
            if tracking.get("assigned_support_agent"):
                st.write(f"**Assigned demo reviewer:** {tracking['assigned_support_agent']}")
            if tracking.get("last_update"):
                st.caption(f"Latest recorded update: {tracking['last_update']}")
            interactions = aria.safe_select("case_interactions", {"case_id": case_id}, limit=100)
            seen_event = next((row for row in reversed(interactions) if row.get("interaction_type") == "SUPPORT_SEEN_DEMO"), None)
            if seen_event:
                st.success("A demo support reviewer has opened this ticket.")

            final_was_reached = any(event.get("event") == "FINAL_DECISION" for event in events)
            if final_was_reached and not pending:
                st.markdown("#### Was this helpful?")
                rating = st.radio("Your feedback", ["Satisfied", "Unsatisfied"], horizontal=True, key=f"rating_{case_id}")
                ask_staff = rating == "Unsatisfied" and st.checkbox("Ask demo customer support to review this ticket", key=f"ask_staff_{case_id}")
                if st.button("Submit feedback", key=f"feedback_{case_id}"):
                    feedback_request = aria.FeedbackRequest(
                        case_id=case_id,
                        satisfied=(rating == "Satisfied"),
                        customer_requested_support=bool(ask_staff),
                        feedback_text="Customer feedback submitted in the public synthetic-data demo.",
                    )
                    identity = {"role": "customer", "customer_id": customer_id, "session_token": "synthetic-demo-session"}
                    try:
                        result = asyncio.run(aria.submit_feedback(feedback_request, identity))
                        if isinstance(result, dict) and result.get("success"):
                            st.success(result.get("message") or "Feedback recorded.")
                            if ask_staff:
                                st.info("The in-app demo support queue now contains this ticket. No real team was notified.")
                        else:
                            st.warning("Feedback could not be confirmed in the demo case store.")
                    except Exception:
                        st.error("Feedback could not be confirmed. The case status was not changed.")


def read_demo_tables() -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    return (
        aria.safe_select("dispute_cases", {}, limit=200),
        aria.safe_select("bra_agent_case_state", {}, limit=200),
        aria.safe_select("support_escalations", {}, limit=200),
        aria.safe_select("case_feedback", {}, limit=500),
        aria.safe_select("bra_agent_runs", {}, limit=500),
        aria.safe_select("bra_agent_tool_calls", {}, limit=500),
    )


def support_view() -> None:
    st.subheader("Customer support")
    st.caption("Synthetic queue · assignment and seen status are recorded in-app; no email or push notification is sent.")
    cases, states, escalations, _feedback, _runs, _tool_calls = read_demo_tables()
    state_map = {str(r.get("case_id")): r for r in states}
    escalation_map = {str(r.get("case_id")): r for r in escalations}
    pending = []
    for case in cases:
        cid = str(case.get("case_id") or "")
        state, escalation = state_map.get(cid, {}), escalation_map.get(cid, {})
        if escalation or str(case.get("case_status") or state.get("state") or "").upper() in {"ESCALATED", "HUMAN_REVIEW", "WAITING_FOR_HUMAN"}:
            pending.append((case, state, escalation))
    if not pending:
        st.info("No cases are waiting for demo support yet. A customer can request staff review from the Customer view.")
        return
    table = [{
        "Ticket": c.get("ticket_id") or s.get("ticket_id") or c.get("case_id"),
        "Issue": c.get("category") or s.get("category") or "Banking query",
        "Status": c.get("case_status") or s.get("state") or e.get("status") or "OPEN",
        "Priority": e.get("priority") or "NORMAL",
        "Assigned demo reviewer": e.get("assigned_support_agent") or "Not assigned",
    } for c, s, e in pending]
    st.dataframe(table, hide_index=True, use_container_width=True)
    tickets = [str(r["Ticket"]) for r in table]
    selected = st.selectbox("Open a case brief", tickets)
    case, state, escalation = next(
        row for row in pending
        if str(row[0].get("ticket_id") or row[1].get("ticket_id") or row[0].get("case_id")) == selected
    )
    st.markdown(f"#### {selected} · {case.get('category') or state.get('category') or 'Banking query'}")
    st.write(f"**Customer request:** {case.get('description') or state.get('description') or 'Not recorded'}")
    st.write(f"**ARIA recommendation:** {state.get('outcome') or case.get('outcome_type') or 'Staff review requested'}")
    st.write(f"**Assigned reviewer:** {escalation.get('assigned_support_agent') or 'Not assigned'}")
    with st.expander("Evidence, checks, and policy trail"):
        for key, label in (("known_facts", "Verified facts"), ("missing_facts", "Still unknown"), ("policy_state", "Policy references"), ("evidence_state", "Evidence status")):
            value = state.get(key)
            if value not in (None, "", [], {}):
                st.write(f"**{label}**")
                st.json(value, expanded=False)
        tool_history = state.get("tool_history")
        if isinstance(tool_history, list):
            st.write(f"**Checks recorded:** {len(tool_history)}")
    if st.button("Mark as seen in this demo", key=f"seen_{case.get('case_id')}"):
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "interaction_id": aria.compact_id("INT", 12),
            "case_id": case.get("case_id"),
            "interaction_type": "SUPPORT_SEEN_DEMO",
            "customer_response": "A demo support reviewer opened this case.",
            "response_text": json.dumps({"demo": True, "seen_at": now}),
            "created_at": now,
        }
        if aria.safe_insert("case_interactions", payload):
            st.success("Seen status recorded for the customer view.")
        else:
            st.error("Seen status could not be confirmed in storage.")


def management_view() -> None:
    st.subheader("Management")
    st.caption("Synthetic portfolio metrics only. These are not actual bank outcomes, revenue, or cost savings.")
    cases, states, escalations, feedback, runs, tool_calls = read_demo_tables()
    ids = {str(r.get("case_id")) for r in cases if r.get("case_id")}
    escalation_count = sum(1 for r in escalations if str(r.get("case_id")) in ids)
    complete_count = sum(1 for r in states if str(r.get("state") or "").upper() in {"RESOLVED", "CLOSED", "INFORMATION_PROVIDED"})
    satisfied_count = sum(1 for r in feedback if r.get("satisfied") is True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demo cases", f"{len(ids):,}")
    c2.metric("Support escalations", f"{escalation_count:,}")
    c3.metric("Completed outcomes", f"{complete_count:,}")
    c4.metric("Feedback responses", f"{len(feedback):,}")
    if feedback:
        c1, _ = st.columns([1, 3])
        c1.metric("Satisfied feedback", f"{round(100 * satisfied_count / len(feedback))}%")
    else:
        st.info("No customer feedback has been recorded in the synthetic demo yet.")
    valid_latencies = [int(row["latency_ms"]) for row in tool_calls if isinstance(row.get("latency_ms"), (int, float)) and row.get("latency_ms") >= 0]
    successful_calls = sum(1 for row in tool_calls if row.get("success") is True)
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Recorded tool calls", f"{len(tool_calls):,}")
    mc2.metric("Tool success rate", f"{round(100 * successful_calls / len(tool_calls))}%" if tool_calls else "No data")
    mc3.metric("Average tool latency", f"{round(sum(valid_latencies) / len(valid_latencies))} ms" if valid_latencies else "No data")

    model_counts: dict[str, int] = {}
    for row in runs:
        model = str(row.get("model_name") or "Not recorded")
        model_counts[model] = model_counts.get(model, 0) + 1
    if model_counts:
        st.markdown("#### Model usage in the demo data")
        st.bar_chart(model_counts)
    st.markdown("#### Product and operating signals")
    st.dataframe([
        {"Signal": "Issue mix and case outcomes", "Source": "Synthetic case records"},
        {"Signal": "Repeat contact and satisfaction", "Source": "Requires linked feedback and lifecycle events"},
        {"Signal": "Provider cost per case", "Source": "Token and cost telemetry is not persisted yet; no cost or savings claim shown"},
        {"Signal": "Potential revenue / savings", "Source": "Requires an approved bank baseline and validated cost model"},
        {"Signal": "Loop and retry controls", "Source": "Configured agent iteration, planner-call, and tool-failure budgets"},
    ], hide_index=True, use_container_width=True)
    with st.expander("How ARIA works (portfolio explanation)"):
        st.markdown(
            "- ARIA chooses checks based on the issue and available evidence, then can re-plan after each result.\n"
            "- RAG retrieves operational guidance; retrieved text is not proof of a customer fact or an official policy decision.\n"
            "- Source-backed policy records and case-scoped bank data determine what ARIA can conclude.\n"
            "- The 9.10.7 candidate compares LangGraph planner selection in shadow mode; the investigation loop remains the existing Python agent.\n"
            "- ARIA recommends financial actions. This portfolio release never executes them."
        )


view = st.radio("ARIA views", ["Customer", "Customer support", "Management"], horizontal=True, label_visibility="collapsed")
if view == "Customer":
    customer_view()
elif view == "Customer support":
    support_view()
else:
    management_view()

st.divider()
st.caption("ARIA portfolio build · synthetic records only · no real support or merchant contact · no financial actions")
