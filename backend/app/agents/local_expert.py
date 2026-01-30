"""
Fixed AI Agent - Now supports both sync and async
Production-ready with better calculations
"""
import os
import re
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key and api_key != "your_key_here":
        genai.configure(api_key=api_key)
        GEMINI_AVAILABLE = True
        print("[OK] Gemini API configured")
except Exception as e:
    print(f"[WARNING] Gemini not available: {e}")

class LocalExpertAgent:
    """Enhanced AI Real Estate Agent"""
    
    def __init__(self):
        self.name = "GeoInsight AI Agent"
        self.version = "3.0.0"
        self.use_gemini = GEMINI_AVAILABLE
    
    def extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text - handles $, K, M"""
        if not text or len(text) > 10000: 
            return []
        
        numbers = []
        
        # More intelligent pattern matching to avoid duplicates
        # Look for numbers with multipliers (K, M) first
        multiplier_pattern = r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*([KkMm])'
        matches = re.finditer(multiplier_pattern, text)
        seen_positions = set()
        
        for match in matches:
            num_str = match.group(1).replace(',', '')
            multiplier = match.group(2).upper()
            
            try:
                num = float(num_str)
                if multiplier == 'K':
                    num *= 1000
                elif multiplier == 'M':
                    num *= 1000000
                if num > 0:  # Only add non-zero numbers
                    numbers.append(num)
                seen_positions.add((match.start(), match.end()))
            except ValueError:
                continue
        
        # Then look for standalone large numbers (5+ digits without multiplier)
        large_number_pattern = r'(?<!\d)(\d{5,})(?!\d)'
        matches = re.finditer(large_number_pattern, text)
        
        for match in matches:
            # Check if this is part of an already processed multiplier match
            if any(match.start() >= start and match.end() <= end for start, end in seen_positions):
                continue
                
            try:
                num = float(match.group(1))
                if num > 0:  # Only add non-zero numbers
                    numbers.append(num)
                seen_positions.add((match.start(), match.end()))
            except ValueError:
                continue
        
        return numbers
    
    def calculate_investment_metrics(
        self,
        price: float,
        monthly_rent: float,
        down_payment_pct: float = 25,
        interest_rate: float = 6.5,
        loan_term: int = 30
    ) -> Dict[str, float]:
        """Calculate comprehensive investment metrics"""
        
        # Annual figures
        annual_rent = monthly_rent * 12
        annual_expenses = annual_rent * 0.35  # 35% rule of thumb
        annual_net_income = annual_rent - annual_expenses
        
        # Loan calculations
        down_payment = price * (down_payment_pct / 100)
        loan_amount = price - down_payment
        
        monthly_rate = interest_rate / 100 / 12
        num_payments = loan_term * 12
        
        if loan_amount > 0 and monthly_rate > 0:
            monthly_mortgage = loan_amount * (
                monthly_rate * (1 + monthly_rate) ** num_payments
            ) / ((1 + monthly_rate) ** num_payments - 1)
        else:
            monthly_mortgage = 0
        
        # Cash flow
        monthly_expenses = annual_expenses / 12
        monthly_cash_flow = monthly_rent - monthly_expenses - monthly_mortgage
        
        # Key metrics
        rental_yield = (annual_rent / price) * 100 if price > 0 else 0
        cap_rate = (annual_net_income / price) * 100 if price > 0 else 0
        cash_on_cash_roi = (annual_net_income / down_payment) * 100 if down_payment > 0 else 0
        
        # Break-even calculation
        break_even_occupancy = (monthly_mortgage / monthly_rent * 100) if monthly_rent > 0 else 100
        
        return {
            'price': price,
            'monthly_rent': monthly_rent,
            'down_payment': down_payment,
            'down_payment_pct': down_payment_pct,
            'loan_amount': loan_amount,
            'interest_rate': interest_rate,
            'monthly_mortgage': monthly_mortgage,
            'monthly_expenses': monthly_expenses,
            'monthly_cash_flow': monthly_cash_flow,
            'annual_rent': annual_rent,
            'annual_expenses': annual_expenses,
            'annual_net_income': annual_net_income,
            'rental_yield': rental_yield,
            'cap_rate': cap_rate,
            'cash_on_cash_roi': cash_on_cash_roi,
            'break_even_occupancy': break_even_occupancy
        }
    
    def format_investment_response(self, metrics: Dict[str, float]) -> str:
        """Format metrics into readable response"""
        
        is_positive = metrics['monthly_cash_flow'] > 0
        
        # Determine quality
        roi = metrics['cash_on_cash_roi']
        if roi > 12:
            quality = "EXCELLENT [5-STAR]"
            recommendation = "[STRONG BUY] Outstanding returns with positive cash flow"
        elif roi > 8:
            quality = "GOOD [4-STAR]"
            recommendation = "[BUY] Solid investment with good returns"
        elif roi > 5:
            quality = "FAIR [3-STAR]"
            recommendation = "[MODERATE] Acceptable but marginal returns"
        else:
            quality = "POOR"
            recommendation = "[AVOID] Returns too low for the risk"
        
        if not is_positive:
            recommendation = "[AVOID] Negative cash flow, not sustainable"
        
        response = f"""**INVESTMENT ANALYSIS REPORT**

**PROPERTY OVERVIEW**
• Purchase Price: **${metrics['price']:,.0f}**
• Monthly Rent: **${metrics['monthly_rent']:,.0f}**
• Down Payment: **{metrics['down_payment_pct']:.0f}%** (${metrics['down_payment']:,.0f})
• Financing: ${metrics['loan_amount']:,.0f} @ {metrics['interest_rate']:.1f}% for 30 years

**MONTHLY CASH FLOW ANALYSIS**
• Rental Income: +${metrics['monthly_rent']:,.0f}
• Operating Expenses (35%): -${metrics['monthly_expenses']:,.0f}
• Mortgage Payment: -${metrics['monthly_mortgage']:,.0f}
• **Net Monthly Cash Flow: ${metrics['monthly_cash_flow']:,.0f}** {'[+]' if is_positive else '[-]'}

**ANNUAL PERFORMANCE**
• Gross Rental Income: ${metrics['annual_rent']:,.0f}
• Total Expenses: ${metrics['annual_expenses']:,.0f}
• Net Operating Income: **${metrics['annual_net_income']:,.0f}**

**KEY INVESTMENT METRICS**
1. **Rental Yield:** {metrics['rental_yield']:.2f}% {'[GOOD]' if metrics['rental_yield'] > 5 else '[LOW]'}
2. **Cap Rate:** {metrics['cap_rate']:.2f}% {'[GOOD]' if metrics['cap_rate'] > 4 else '[LOW]'}
3. **Cash-on-Cash ROI:** {metrics['cash_on_cash_roi']:.2f}% {'[EXCELLENT]' if metrics['cash_on_cash_roi'] > 10 else '[GOOD]' if metrics['cash_on_cash_roi'] > 6 else '[FAIR]'}
4. **Break-Even Occupancy:** {metrics['break_even_occupancy']:.1f}%

**INVESTMENT QUALITY: {quality}**

**RECOMMENDATION:**
{recommendation}

**ADDITIONAL CONSIDERATIONS:**
• Property taxes and insurance not included (add ~1.5% of property value annually)
• Maintenance reserve recommended: $50-100/month
• Vacancy factor: Budget for 5-10% annual vacancy
• Consider appreciation: Typical 3-5% annually in growing markets

**DECISION FACTORS:**
[+] Buy if: Positive cash flow + ROI > 8% + growing market
[!] Reconsider if: Negative cash flow OR ROI < 6%
[-] Avoid if: High vacancy risk + negative cash flow
"""
        return response
    
    async def query_gemini(self, query: str, context: str = "") -> str:
        """Query Gemini API"""
        if not self.use_gemini:
            return None
        
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = f"""You are a professional real estate investment advisor. Provide clear, actionable advice based on market data and investment principles.

User Query: {query}

{context}

Provide a concise, professional response with specific recommendations. Focus on numbers and concrete advice."""
            
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini error: {e}")
            return None
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main query processor - now async compatible
        Handles both sync and async calls
        """
        
        query_lower = query.lower()
        numbers = self.extract_numbers(query)
        
        # INVESTMENT ANALYSIS
        if any(word in query_lower for word in ['investment', 'roi', 'calculate', 'analyze', 'cash flow']):
            if len(numbers) >= 2:
                price = numbers[0]
                monthly_rent = numbers[1]
                
                # Extract down payment if specified
                down_match = re.search(r'(\d+)%?\s*down', query_lower)
                down_payment_pct = float(down_match.group(1)) if down_match else 25
                
                metrics = self.calculate_investment_metrics(price, monthly_rent, down_payment_pct)
                answer = self.format_investment_response(metrics)
                
                return {
                    'query': query,
                    'answer': answer,
                    'calculations': metrics,
                    'success': True,
                    'confidence': 0.95,
                    'type': 'investment_analysis'
                }
        
        # PRICE ANALYSIS
        if any(word in query_lower for word in ['price', 'worth', 'value', 'cost']):
            if numbers:
                price = numbers[0]
                
                # Try Gemini first
                if self.use_gemini:
                    context = f"Property price: ${price:,.0f}"
                    gemini_response = await self.query_gemini(query, context)
                    if gemini_response:
                        return {
                            'query': query,
                            'answer': gemini_response,
                            'price_analyzed': price,
                            'success': True,
                            'confidence': 0.9,
                            'source': 'gemini',
                            'type': 'price_analysis'
                        }
                
                # Fallback
                answer = f"""**PRICE ANALYSIS**

Price: **${price:,.0f}**

**Market Context:**
• Low-tier market: $200K-350K (starter homes, condos)
• Mid-tier market: $350K-600K (family homes, good neighborhoods)
• High-tier market: $600K+ (luxury, premium locations)

Your price falls in: **{
    'Low-tier' if price < 350000 else
    'Mid-tier' if price < 600000 else
    'High-tier'
}**

**Expected Property Features:**
• Size: ~{int(price/250)} sq ft
• Bedrooms: {'1-2' if price < 300000 else '2-3' if price < 500000 else '3-4'}
• Bathrooms: {'1-1.5' if price < 300000 else '2-2.5' if price < 500000 else '2.5-3.5'}
• Condition: {'Basic' if price < 300000 else 'Good' if price < 500000 else 'Excellent'}

**Price per Sq Ft Guide:**
• Economy: $100-150/sqft
• Standard: $150-250/sqft  
• Premium: $250-400/sqft
• Luxury: $400+/sqft

**Your estimate: ${price / int(price/250):.0f}/sqft**

**Recommendation:** Compare with recent sales (comps) in the area for accurate valuation."""
                
                return {
                    'query': query,
                    'answer': answer,
                    'price_analyzed': price,
                    'success': True,
                    'confidence': 0.85,
                    'type': 'price_analysis'
                }
        
        # RENTAL ANALYSIS
        if any(word in query_lower for word in ['rent', 'rental', 'lease']):
            if numbers:
                rent = numbers[0]
                suggested_value = rent * 100  # 1% rule
                
                answer = f"""**RENTAL MARKET ANALYSIS**

Monthly Rent: **${rent:,.0f}**

**Market Positioning:**
• Studio: $1,200-1,800 {'✅ Competitive' if 1200 <= rent <= 1800 else ''}
• 1-Bedroom: $1,800-2,400 {'✅ Competitive' if 1800 <= rent <= 2400 else ''}
• 2-Bedroom: $2,400-3,000 {'✅ Competitive' if 2400 <= rent <= 3000 else ''}
• 3-Bedroom: $3,000-3,800 {'✅ Competitive' if 3000 <= rent <= 3800 else ''}

**1% RULE CHECK**
Property value should be ~100x monthly rent
• Suggested value: **${suggested_value:,.0f}**
• Acceptable range: ${suggested_value*0.8:,.0f} - ${suggested_value*1.2:,.0f}

**LANDLORD METRICS**
• Monthly expenses (35%): ${rent * 0.35:,.0f}
• Net income: ${rent * 0.65:,.0f}
• Target rental yield: 4-8% annually
• Minimum ROI: 6% cash-on-cash

**TENANT AFFORDABILITY**
Rule of thumb: Rent ≤ 30% of gross income
• Minimum income needed: ${rent * 3:,.0f}/month (${rent * 36:,.0f}/year)

**Verdict:** {'✅ Fair rent' if 1500 <= rent <= 3500 else '⚠️ Verify market rates'}"""
                
                return {
                    'query': query,
                    'answer': answer,
                    'rent_analyzed': rent,
                    'success': True,
                    'confidence': 0.88,
                    'type': 'rental_analysis'
                }
        
        # GENERAL QUERY - Try Gemini
        if self.use_gemini:
            try:
                gemini_response = await self.query_gemini(query)
                if gemini_response:
                    return {
                        'query': query,
                        'answer': gemini_response,
                        'success': True,
                        'confidence': 0.9,
                        'source': 'gemini',
                        'type': 'general'
                    }
            except Exception as e:
                print(f"Gemini query error: {e}")
        
        # FALLBACK - More helpful generic response
        return {
            'query': query,
            'answer': f"""**GeoInsight AI Assistant**

I'm your real estate intelligence assistant. I can help with:

**📊 Investment Analysis:**
- "Calculate ROI for 9000000 property with 15000 monthly rent"
- "Analyze investment: 7000000 price, 9000 rent, 20% down"

**💰 Price Analysis:**
- "Is 450000 a good price for a 3BHK?"
- "What's fair market value for 750K property?"

**🏠 Rental Analysis:**
- "Fair rent for 400K property?"
- "Expected rent for 350K house"

**🏙️ Market Insights:**
- "Best neighborhoods for rental investment?"
- "Compare downtown vs suburban properties"

For general knowledge questions (like population, weather, etc), I'm optimized for real estate analysis. Try asking me about property investments instead!""",
            'success': True,
            'confidence': 0.7,
            'type': 'help'
        }

# Global instance
agent = LocalExpertAgent()