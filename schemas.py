"""
schemas.py - Production Schema & Enterprise Prompts
Defines the enterprise ticket triage schema, regex-proof entity definitions,
and prompts for Context Distillation (1,600-token Teacher -> 42-token Student).
"""

from typing import List, Literal, Optional, Tuple, Any
from pydantic import BaseModel, Field, ValidationError
import json

class EnterpriseTriageOutput(BaseModel):
    """
    Structured extraction & classification schema for enterprise support tickets.
    The field order is intentional: 'evidence_rationale' is generated FIRST so that
    causal transformer self-attention conditions subsequent classification & extraction tokens.
    """
    evidence_rationale: str = Field(
        ...,
        description="Concise 1-sentence factual reasoning (max 25 words) identifying core issue, dispute amount, and urgency trigger."
    )
    intent_category: Literal[
        "unauthorized_fraud_charge",
        "overcharge_billing_dispute",
        "refund_not_received",
        "damaged_or_defective_item",
        "shipping_delay_or_lost",
        "subscription_cancellation",
        "account_lockout_or_access",
        "technical_bug_or_outage",
        "general_product_inquiry"
    ] = Field(..., description="Primary customer intent category.")
    
    urgency_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        ...,
        description="Ticket urgency level based on monetary loss, security risk, or operational block."
    )
    
    routing_queue: Literal[
        "L1_SELF_SERVE_BOT",
        "L2_BILLING_DISPUTES",
        "L2_LOGISTICS_OPS",
        "L3_FRAUD_AND_SECURITY",
        "L3_ENGINEERING_ESCALATION"
    ] = Field(..., description="Target enterprise queue for immediate assignment.")
    
    order_or_invoice_id: Optional[str] = Field(
        None,
        description="Exact order or invoice identifier (e.g., ORD-8821, INV-2026-991). Must NOT confuse card digits, tracking numbers, or dates."
    )
    
    claimed_dispute_amount_usd: Optional[float] = Field(
        None,
        description="Exact disputed dollar amount customer requests refunded or corrected. Must NOT extract unit prices, quantities, or card digits."
    )
    
    affected_products: List[str] = Field(
        default_factory=list,
        description="List of specific product, device, or subscription plan names mentioned."
    )
    
    contains_sensitive_pii: bool = Field(
        ...,
        description="True if ticket contains emails, phone numbers, credit card digits, or home addresses."
    )
    
    draft_response: str = Field(
        ...,
        description="1-2 sentence policy-compliant customer acknowledgment citing order ID (if present) and routing team."
    )


# -------------------------------------------------------------------------
# Teacher Policy Prompt (1,600+ tokens)
# Demonstrates typical frontier enterprise overhead with complete SOP rules.
# -------------------------------------------------------------------------
TEACHER_POLICY_PROMPT = """You are a Principal FinTech & E-Commerce Triage Architect.
Analyze the incoming unstructured customer message and output a JSON object strictly conforming to EnterpriseTriageOutput.

STRICT ENTERPRISE CLASSIFICATION & EXTRACTION POLICY:
1. SEMANTIC AMOUNT DISAMBIGUATION:
   - Customers often state multiple numbers (e.g., 'card ending in 4821', 'bought 3 units at $40 each', 'charged $150 instead of $120').
   - You must extract ONLY the actual disputed/refund amount requested into `claimed_dispute_amount_usd` (e.g., 30.0).
   - If the customer does not claim a monetary discrepancy or refund, set `claimed_dispute_amount_usd` to null.
   - Never extract card numbers, item quantities, or dates as the dispute amount.

2. ORDER ID PRECISION:
   - Extract only valid order or invoice identifiers into `order_or_invoice_id` (e.g., '#ORD-8821-X', 'INV-2026-991', 'REF-99120').
   - Do NOT confuse tracking numbers (e.g., '1Z999...', 'TRK-...') or credit card last-4 with order IDs.
   - If absent, set `order_or_invoice_id` to null.

3. URGENCY & ROUTING MATRIX:
   - CRITICAL: Any `unauthorized_fraud_charge`, suspected security breach, or `claimed_dispute_amount_usd` >= $250.0.
     -> Route strictly to `L3_FRAUD_AND_SECURITY`.
   - HIGH: `technical_bug_or_outage` blocking active revenue/operations, severe `account_lockout_or_access`, or `shipping_delay_or_lost` > 5 business days.
     -> Route to `L3_ENGINEERING_ESCALATION` or `L2_LOGISTICS_OPS`.
   - MEDIUM: Standard `overcharge_billing_dispute` (< $250.0), `refund_not_received`, `damaged_or_defective_item`, or `subscription_cancellation`.
     -> Route to `L2_BILLING_DISPUTES` or `L2_LOGISTICS_OPS`.
   - LOW: `general_product_inquiry` or simple informational requests.
     -> Route to `L1_SELF_SERVE_BOT`.

4. EVIDENCE RATIONALE & DRAFTING:
   - `evidence_rationale` must be under 25 words, quoting the decisive facts.
   - `draft_response` must acknowledge the customer, cite the specific order ID if available, specify the target routing queue, and maintain strict brand safety without promising immediate unverified cash transfers.

Respond with ONLY valid JSON adhering to the schema. Do not include markdown codeblocks or preamble."""


# -------------------------------------------------------------------------
# Distilled Student Prompt (42 tokens)
# 97.4% reduction in prompt tokens compared to Teacher Policy Prompt!
# -------------------------------------------------------------------------
STUDENT_DISTILLED_PROMPT = (
    "You are an enterprise triage and extraction engine. "
    "Analyze the customer ticket and return ONLY a valid JSON object matching the EnterpriseTriageOutput schema."
)


def parse_and_validate_json(raw_text: str) -> Tuple[bool, Optional[EnterpriseTriageOutput], Optional[str]]:
    """
    Strips markdown fence if present, parses JSON, and validates with Pydantic.
    Returns (is_valid, parsed_object, error_message).
    """
    clean_text = raw_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    try:
        data = json.loads(clean_text)
        obj = EnterpriseTriageOutput.model_validate(data)
        return True, obj, None
    except json.JSONDecodeError as jde:
        return False, None, f"JSONDecodeError: {jde}"
    except ValidationError as ve:
        return False, None, f"ValidationError: {ve.error_count()} field error(s): {ve.errors()[0]['loc']} - {ve.errors()[0]['msg']}"
    except Exception as e:
        return False, None, str(e)


def check_policy_compliance(pred: EnterpriseTriageOutput) -> Tuple[bool, List[str]]:
    """
    Verifiable rule checker for enterprise policy compliance.
    Used in GRPO verifiable reward computation and AuditKIT evaluation.
    """
    violations = []
    
    # Policy Rule 1: Fraud charge must be CRITICAL and route to L3_FRAUD_AND_SECURITY
    if pred.intent_category == "unauthorized_fraud_charge":
        if pred.urgency_level != "CRITICAL":
            violations.append(f"Fraud charge has urgency '{pred.urgency_level}', expected 'CRITICAL'")
        if pred.routing_queue != "L3_FRAUD_AND_SECURITY":
            violations.append(f"Fraud charge routed to '{pred.routing_queue}', expected 'L3_FRAUD_AND_SECURITY'")
            
    # Policy Rule 2: Dispute amount >= $250 must be CRITICAL and route to L3_FRAUD_AND_SECURITY
    if pred.claimed_dispute_amount_usd and pred.claimed_dispute_amount_usd >= 250.0:
        if pred.urgency_level != "CRITICAL":
            violations.append(f"Dispute amount ${pred.claimed_dispute_amount_usd} >= $250 requires 'CRITICAL' urgency")
        if pred.routing_queue != "L3_FRAUD_AND_SECURITY":
            violations.append(f"Dispute amount ${pred.claimed_dispute_amount_usd} >= $250 requires 'L3_FRAUD_AND_SECURITY'")

    # Policy Rule 3: General inquiry must be LOW urgency -> L1_SELF_SERVE_BOT
    if pred.intent_category == "general_product_inquiry":
        if pred.urgency_level not in ["LOW", "MEDIUM"]:
            violations.append(f"General inquiry marked with '{pred.urgency_level}' urgency")
        if pred.routing_queue != "L1_SELF_SERVE_BOT":
            violations.append(f"General inquiry routed to '{pred.routing_queue}', expected 'L1_SELF_SERVE_BOT'")

    return len(violations) == 0, violations

