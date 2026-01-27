"""
AI Assistant Page
Natural language interface for real estate queries
"""
import streamlit as st
from api_client import api
from utils import (
    format_currency, format_number, format_percentage,
    get_roi_label, show_success_message, show_error_message,
    get_session_state, set_session_state
)
from components.header import render_section_header
from datetime import datetime

def render_ai_assistant_page():
    """Main AI assistant page"""
    render_section_header("AI Real Estate Assistant", "🤖")
    
    st.markdown("Get instant investment analysis, property valuations, and market insights powered by AI.")
    
    # Example queries
    render_example_queries()
    
    # Main query interface
    render_query_interface()
    
    # History toggle
    if st.session_state.get('show_ai_history', False):
        st.divider()
        render_query_history()

def render_example_queries():
    """Show example queries in expandable section"""
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **📊 Investment Analysis:**
        - `Calculate ROI for $300,000 property with $2,000 monthly rent`
        - `Investment analysis: $500K property, $2,800/month rent, 20% down`
        - `Analyze cash flow for $450K house with $2,500 rent`
        
        **💰 Property Valuation:**
        - `Is $450,000 a good price for a 3-bedroom house?`
        - `What's fair market value for $750K property?`
        - `Price analysis for $600K condo`
        
        **🏠 Rental Analysis:**
        - `Fair rent for $400K property?`
        - `Rental market for $2,500/month apartment`
        - `Expected rent for $350K house`
        
        **🏙️ Market Insights:**
        - `Best neighborhoods for rental investment?`
        - `Compare downtown vs suburban properties`
        - `Appreciation trends in tech hubs`
        """)

def render_query_interface():
    """Render the main query input and response area"""
    # Pre-fill from properties page if navigated
    default_query = get_session_state('ai_query', '')
    
    query = st.text_area(
        "💬 Ask Your Question",
        value=default_query,
        placeholder="e.g., Calculate ROI for $300,000 property with $2,000 monthly rent",
        height=100,
        key="ai_assistant_query"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        ask_button = st.button(
            "🚀 Ask AI Assistant", 
            type="primary", 
            use_container_width=True,
            key="ask_ai"
        )
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            set_session_state('ai_query', '')
            st.rerun()
    
    with col3:
        if st.button("📜 History", use_container_width=True):
            current = st.session_state.get('show_ai_history', False)
            st.session_state.show_ai_history = not current
            st.rerun()
    
    if ask_button and query:
        handle_query_submission(query)
    elif ask_button:
        show_error_message("Please enter a question")

def handle_query_submission(query: str):
    """Handle AI query submission"""
    with st.spinner("🤔 AI analyzing..."):
        response = api.query_ai_agent(query)
    
    if not response:
        return
    
    if not response.get('success'):
        show_error_message("AI query failed")
        return
    
    # Save to history
    history = get_session_state('agent_history', [])
    history.append({
        'query': query,
        'response': response,
        'timestamp': datetime.now()
    })
    st.session_state.agent_history = history[-20:]  # Keep last 20
    
    show_success_message("Analysis Complete")
    
    # Display response
    render_ai_response(response)

def render_ai_response(response: dict):
    """Display AI response with formatting"""
    st.markdown("### 💡 AI Response")
    
    answer = response.get('response', {}).get('answer', '')
    st.markdown(answer)
    
    # Show calculations if available
    calculations = response.get('response', {}).get('calculations')
    
    if calculations:
        render_investment_breakdown(calculations)
    
    # Confidence score
    confidence = response.get('confidence', 0)
    if confidence:
        render_confidence_score(confidence)

def render_investment_breakdown(calculations: dict):
    """Render detailed investment calculations"""
    st.divider()
    st.subheader("📊 Investment Breakdown")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price = calculations.get('price', 0)
        st.metric("🏠 Price", format_currency(price))
    
    with col2:
        rent = calculations.get('monthly_rent', 0)
        st.metric("💵 Rent/Mo", format_currency(rent))
    
    with col3:
        cash_flow = calculations.get('monthly_cash_flow', 0)
        delta_color = "normal" if cash_flow > 0 else "inverse"
        st.metric("💰 Cash Flow", format_currency(cash_flow))
    
    with col4:
        roi = calculations.get('cash_on_cash_roi', 0)
        label, emoji = get_roi_label(roi)
        st.metric("📈 ROI", f"{roi:.1f}%", help=f"{label} {emoji}")
    
    # Detailed breakdown in expander
    with st.expander("📋 Full Financial Details"):
        render_financial_details(calculations)

def render_financial_details(calc: dict):
    """Render detailed financial breakdown"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💰 Investment**")
        st.write(f"Down Payment: {format_currency(calc.get('down_payment', 0))}")
        st.write(f"Down %: {calc.get('down_payment_pct', 0):.0f}%")
        st.write(f"Loan Amount: {format_currency(calc.get('loan_amount', 0))}")
        st.write(f"Interest Rate: {calc.get('interest_rate', 0):.1f}%")
    
    with col2:
        st.markdown("**📊 Monthly**")
        st.write(f"Mortgage: {format_currency(calc.get('monthly_mortgage', 0))}")
        st.write(f"Expenses: {format_currency(calc.get('monthly_expenses', 0))}")
        st.write(f"Net Income: {format_currency(calc.get('monthly_cash_flow', 0))}")
    
    st.markdown("**📈 Annual**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"Gross Rent: {format_currency(calc.get('annual_rent', 0))}")
    
    with col2:
        st.write(f"Expenses: {format_currency(calc.get('annual_expenses', 0))}")
    
    with col3:
        st.write(f"NOI: {format_currency(calc.get('annual_net_income', 0))}")
    
    st.markdown("**🎯 Key Ratios**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"Rental Yield: {format_percentage(calc.get('rental_yield', 0))}")
    
    with col2:
        st.write(f"Cap Rate: {format_percentage(calc.get('cap_rate', 0))}")
    
    with col3:
        st.write(f"Break-Even: {format_percentage(calc.get('break_even_occupancy', 0))}")

def render_confidence_score(confidence: float):
    """Render AI confidence score"""
    st.divider()
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.progress(confidence)
    
    with col2:
        st.metric("🎯 Confidence", format_percentage(confidence * 100, decimals=0))

def render_query_history():
    """Render query history"""
    st.subheader("📜 Query History")
    
    history = get_session_state('agent_history', [])
    
    if not history:
        st.info("No query history yet. Ask a question to get started!")
        return
    
    for idx, item in enumerate(reversed(history)):
        render_history_item(item, idx)

def render_history_item(item: dict, idx: int):
    """Render individual history item"""
    query = item['query']
    timestamp = item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    
    with st.expander(f"🤖 {query[:60]}{'...' if len(query) > 60 else ''}"):
        st.caption(f"**Asked:** {timestamp}")
        
        response = item['response']
        answer = response.get('response', {}).get('answer', '')
        
        # Show truncated answer
        if len(answer) > 300:
            st.markdown(answer[:300] + '...')
            if st.button("📖 Show Full Response", key=f"expand_{idx}"):
                st.markdown(answer)
        else:
            st.markdown(answer)
        
        # Quick metrics if available
        calc = response.get('response', {}).get('calculations')
        if calc:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                roi = calc.get('cash_on_cash_roi', 0)
                st.metric("ROI", f"{roi:.1f}%")
            
            with col2:
                flow = calc.get('monthly_cash_flow', 0)
                st.metric("Cash Flow", format_currency(flow))
            
            with col3:
                price = calc.get('price', 0)
                st.metric("Price", format_currency(price))
        
        # Reuse query button
        if st.button("🔄 Ask Again", key=f"reuse_{idx}", use_container_width=True):
            set_session_state('ai_query', query)
            st.rerun()