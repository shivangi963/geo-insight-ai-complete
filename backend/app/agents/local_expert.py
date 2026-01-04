import os
import re
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv


load_dotenv()

class LocalExpertAgent:
    def __init__(self):
        self.name = "GeoInsight AI Agent"
        self.version = "2.0.0"
    
    def process_query(self, query: str) -> Dict[str, Any]:
     
        query_lower = query.lower()
        
        if self._is_investment_query(query_lower) and self._has_numbers(query):
            return self._handle_detailed_investment_query(query)
        
     
        elif self._is_price_query(query_lower) and self._has_numbers(query):
            return self._handle_detailed_price_query(query)
        
      
        elif self._is_rental_query(query_lower) and self._has_numbers(query):
            return self._handle_detailed_rental_query(query)
        
   
        elif any(word in query_lower for word in ["price", "cost", "expensive", "cheap", "value"]):
            return self._handle_price_query(query)
        elif any(word in query_lower for word in ["investment", "roi", "return", "profit", "yield"]):
            return self._handle_investment_query(query)
        elif any(word in query_lower for word in ["location", "area", "neighborhood", "city"]):
            return self._handle_location_query(query)
        elif any(word in query_lower for word in ["rent", "rental", "lease"]):
            return self._handle_rental_query(query)
        elif any(word in query_lower for word in ["amenities", "school", "park", "shop", "transport"]):
            return self._handle_amenities_query(query)
        elif any(word in query_lower for word in ["hello", "hi", "hey", "greetings"]):
            return self._handle_greeting(query)
        else:
            return self._handle_general_query(query)
    
    def _extract_numbers(self, text: str) -> List[float]:
   
        numbers = []
      
        patterns = [
            r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  
            r'\$?(\d+(?:\.\d+)?)\s*(?:k|K)',       
            r'\$?(\d+(?:\.\d+)?)\s*(?:m|M)',        
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match:
                   
                    clean_num = match.replace(',', '')
                    try:
                        num = float(clean_num)
                      
                        if 'k' in text.lower() or 'K' in text.lower():
                            num *= 1000
                        elif 'm' in text.lower() or 'M' in text.lower():
                            num *= 1000000
                        numbers.append(num)
                    except ValueError:
                        continue
        
        return numbers
    
    def _has_numbers(self, text: str) -> bool:
        return len(self._extract_numbers(text)) > 0
    
    def _is_investment_query(self, text: str) -> bool:
        investment_keywords = ["investment", "roi", "return", "profit", "yield", 
                              "cash flow", "cap rate", "calculate", "metrics"]
        return any(keyword in text for keyword in investment_keywords)
    
    def _is_price_query(self, text: str) -> bool:
        price_keywords = ["price", "cost", "value", "worth", "appraisal", 
                         "how much", "valuation"]
        return any(keyword in text for keyword in price_keywords)
    
    def _is_rental_query(self, text: str) -> bool:
        rental_keywords = ["rent", "rental", "lease", "renting", "tenant"]
        return any(keyword in text for keyword in rental_keywords)
    
    def _handle_detailed_investment_query(self, query: str) -> Dict[str, Any]:
        numbers = self._extract_numbers(query)
        
       
        price = 250000
        monthly_rent = 1800
        down_payment_percent = 25 
        loan_term = 30  
        interest_rate = 6.5 
        
     
        if len(numbers) >= 1:
            price = numbers[0]
        if len(numbers) >= 2:
            monthly_rent = numbers[1]
        
       
        dp_match = re.search(r'(\d+)%\s*down', query.lower())
        if dp_match:
            down_payment_percent = float(dp_match.group(1))
        
    
        annual_rent = monthly_rent * 12
        annual_expenses = annual_rent * 0.35  
        annual_net_income = annual_rent - annual_expenses
        
        down_payment = price * (down_payment_percent / 100)
        loan_amount = price - down_payment
        
        monthly_interest_rate = interest_rate / 100 / 12
        num_payments = loan_term * 12
        if loan_amount > 0:
            monthly_mortgage = loan_amount * (monthly_interest_rate * (1 + monthly_interest_rate) ** num_payments) / ((1 + monthly_interest_rate) ** num_payments - 1)
        else:
            monthly_mortgage = 0
        
     
        monthly_expenses = annual_expenses / 12
        monthly_cash_flow = monthly_rent - monthly_expenses - monthly_mortgage
        
        rental_yield = (annual_rent / price) * 100
        cap_rate = (annual_net_income / price) * 100
        cash_on_cash_roi = (annual_net_income / down_payment) * 100 if down_payment > 0 else 0
        
        # 
        answer = f"""** DETAILED INVESTMENT ANALYSIS**

**PROPERTY SPECIFICS:**
- Purchase Price: **${price:,.0f}**
- Monthly Rent: **${monthly_rent:,.0f}**
- Down Payment: **{down_payment_percent}%** (${down_payment:,.0f})
- Loan Amount: ${loan_amount:,.0f} ({interest_rate}% for {loan_term} years)

**ANNUAL FINANCIALS:**
- Gross Rental Income: **${annual_rent:,.0f}**
- Operating Expenses (35%): ${annual_expenses:,.0f}
- Net Operating Income: **${annual_net_income:,.0f}**
- Annual Mortgage Payments: ${monthly_mortgage * 12:,.0f}

**KEY METRICS:**
1. **Monthly Cash Flow:** ${monthly_cash_flow:,.0f} ({'Positive' if monthly_cash_flow > 0 else ' Negative'})
2. **Rental Yield:** {rental_yield:.1f}% ({'Good' if rental_yield > 5 else ' Below Average'})
3. **Cap Rate:** {cap_rate:.1f}% ({'Good' if cap_rate > 4 else 'Below Average'})
4. **Cash-on-Cash ROI:** {cash_on_cash_roi:.1f}% ({' Excellent' if cash_on_cash_roi > 10 else ' Good' if cash_on_cash_roi > 8 else ' Marginal'})

**RECOMMENDATION:**
{'**STRONG INVESTMENT** - Positive cash flow and good returns' if monthly_cash_flow > 200 and cash_on_cash_roi > 8 else 
'**MODERATE INVESTMENT** - Needs careful management' if monthly_cash_flow > 0 else 
'**POOR INVESTMENT** - Negative cash flow, reconsider'}

**Next Steps:** Consider property taxes, maintenance costs, and vacancy rates for more accurate analysis."""
        
        return {
            "query": query,
            "answer": answer,
            "calculations": {
                "property_price": price,
                "monthly_rent": monthly_rent,
                "down_payment": down_payment,
                "loan_amount": loan_amount,
                "monthly_cash_flow": round(monthly_cash_flow, 2),
                "rental_yield": round(rental_yield, 2),
                "cap_rate": round(cap_rate, 2),
                "cash_on_cash_roi": round(cash_on_cash_roi, 2),
                "annual_net_income": round(annual_net_income, 2)
            },
            "success": True,
            "confidence": 0.92
        }
    
    def _handle_detailed_price_query(self, query: str) -> Dict[str, Any]:
        numbers = self._extract_numbers(query)
        
        if not numbers:
            return self._handle_price_query(query)
        
        price = numbers[0]
        
        answer = f"""** PRICE ANALYSIS**

Your mentioned price: **${price:,.0f}**

**MARKET COMPARISON:**
- Compared to Urban Average ($300K-$800K): {' Within range' if 300000 <= price <= 800000 else ' Below average' if price < 300000 else ' Above average'}
- Compared to Suburban Average ($200K-$500K): {' Within range' if 200000 <= price <= 500000 else ' Below average' if price < 200000 else ' Above average'}

**VALUE ASSESSMENT:**
- For ${price:,.0f}, you should expect:
  • {f'{price/1500:.0f} sq ft' if price <= 500000 else f'{price/2000:.0f} sq ft'} of living space
  • {'2-3 bedrooms, 1-2 bathrooms' if price <= 350000 else '3-4 bedrooms, 2-3 bathrooms' if price <= 600000 else '4+ bedrooms, 3+ bathrooms'}
  • {'Basic amenities' if price <= 300000 else 'Moderate amenities' if price <= 500000 else 'Premium amenities'}

**PRICE TIPS:**
1. Verify recent comparable sales in the area
2. Factor in renovation costs if needed
3. Consider future appreciation (3-5% annually)
4. Negotiation range: ±5-10% of asking price"""
        
        return {
            "query": query,
            "answer": answer,
            "analysis": {
                "price": price,
                "urban_comparison": "Within range" if 300000 <= price <= 800000 else "Below average" if price < 300000 else "Above average",
                "suburban_comparison": "Within range" if 200000 <= price <= 500000 else "Below average" if price < 200000 else "Above average",
                "expected_sqft": round(price / 1800),  
                "expected_bedrooms": "2-3" if price <= 350000 else "3-4" if price <= 600000 else "4+"
            },
            "success": True,
            "confidence": 0.88
        }
    
    def _handle_detailed_rental_query(self, query: str) -> Dict[str, Any]:
        
        numbers = self._extract_numbers(query)
        
        if not numbers:
            return self._handle_rental_query(query)
        
        rent_amount = numbers[0]
        
        suggested_value = rent_amount * 100  
        
        answer = f"""**RENTAL ANALYSIS**

Monthly Rent: **${rent_amount:,.0f}**

**MARKET POSITIONING:**
- Studio Apartments ($1,200-$1,800): {'Competitive' if 1200 <= rent_amount <= 1800 else ' Outside range'}
- 1-Bedroom ($1,800-$2,400): {'Competitive' if 1800 <= rent_amount <= 2400 else ' Outside range'}
- 2-Bedroom ($2,400-$3,000): {' Competitive' if 2400 <= rent_amount <= 3000 else ' Outside range'}
- 3-Bedroom ($3,000-$3,800): {' Competitive' if 3000 <= rent_amount <= 3800 else ' Outside range'}

**PROPERTY VALUE (1% RULE):**
This rent suggests a property value of **${suggested_value:,.0f}**
- Monthly rent should be 0.8-1% of property value
- Actual range: ${suggested_value*0.8:,.0f} to ${suggested_value*1.2:,.0f}

**INVESTOR PERSPECTIVE:**
- For landlords: Target 4-6% rental yield
- Monthly expenses estimate: ${rent_amount * 0.35:,.0f} (35% of rent)
- Net monthly income: ${rent_amount * 0.65:,.0f}
- Break-even occupancy: {100 * (rent_amount * 0.35) / rent_amount:.0f}%

**RECOMMENDATIONS:**
1. Compare with similar properties in the area
2. Factor in amenities, location, and condition
3. Consider seasonal demand variations
4. Review local rental regulations"""
        
        return {
            "query": query,
            "answer": answer,
            "analysis": {
                "rent_amount": rent_amount,
                "suggested_property_value": suggested_value,
                "studio_comparison": "Competitive" if 1200 <= rent_amount <= 1800 else "High" if rent_amount > 1800 else "Low",
                "one_bedroom_comparison": "Competitive" if 1800 <= rent_amount <= 2400 else "High" if rent_amount > 2400 else "Low",
                "estimated_monthly_expenses": rent_amount * 0.35,
                "estimated_net_income": rent_amount * 0.65
            },
            "success": True,
            "confidence": 0.86
        }
    
    def _handle_price_query(self, query: str) -> Dict[str, Any]:
        return {
            "query": query,
            "answer": "Based on current market data, average property prices in urban areas range from $300,000 to $800,000. For suburban areas, expect $200,000 to $500,000. Premium locations can exceed $1 million.",
            "confidence": 0.85,
            "suggestions": [
                "Provide specific location for accurate pricing",
                "Consider property type and size"
            ],
            "success": True
        }
    
    def _handle_investment_query(self, query: str) -> Dict[str, Any]:
        return {
            "query": query,
            "answer": "Investment analysis: Properties typically show 5-8% annual ROI. Consider 20-30% down payment, 4-6% rental yield, and 3-5% annual appreciation. Good investments have positive cash flow after expenses.",
            "confidence": 0.87,
            "calculations": {
                "average_roi": "5-8% annually",
                "rental_yield": "4-6%",
                "appreciation": "3-5% yearly"
            },
            "success": True
        }
    
    def _handle_location_query(self, query: str) -> Dict[str, Any]:
        return {
            "query": query,
            "answer": "Location is key! Properties near schools (+15% value), public transit (+10%), and commercial centers (+20%) have higher appreciation. Safe neighborhoods with low crime rates show 5-10% premium.",
            "confidence": 0.9,
            "metrics": {
                "school_proximity": "+15% value",
                "transit_access": "+10% value",
                "commercial_nearby": "+20% value",
                "safety_premium": "5-10%"
            },
            "success": True
        }
    
    def _handle_rental_query(self, query: str) -> Dict[str, Any]:
        return {
            "query": query,
            "answer": "Rental market: Studios rent for $1,200-$1,800, 1-bedroom $1,800-$2,400, 2-bedroom $2,400-$3,000. Near universities commands 25% premium. Properties should rent for 0.8-1% of property value monthly.",
            "confidence": 0.88,
            "rental_ranges": {
                "studio": "$1,200-$1,800",
                "1_bedroom": "$1,800-$2,400",
                "2_bedroom": "$2,400-$3,000",
                "3_bedroom": "$3,000-$3,800"
            },
            "success": True
        }
    
    def _handle_amenities_query(self, query: str) -> Dict[str, Any]:
        return {
            "query": query,
            "answer": "Amenities impact: Walkability adds 10-20% value. Parks within 1km add 8% premium. Top-rated school districts add 12-15% to property values. Properties with parking spots add 5-10% value.",
            "confidence": 0.86,
            "premiums": {
                "walkability": "10-20%",
                "park_proximity": "8%",
                "top_schools": "12-15%",
                "parking": "5-10%"
            },
            "success": True
        }
    
    def _handle_greeting(self, query: str) -> Dict[str, Any]:
        return {
            "query": query,
            "answer": "Hello! I'm your GeoInsight AI assistant, specialized in real estate analysis. I can help with price insights, investment calculations, location evaluation, rental market analysis, and amenity assessments.",
            "confidence": 0.95,
            "success": True
        }
    
    def _handle_general_query(self, query: str) -> Dict[str, Any]:
        return {
            "query": query,
            "answer": f"I'm your GeoInsight AI real estate expert. You asked: '{query}'. I can analyze property prices, calculate investment returns, evaluate locations, assess rental potential, and compare amenities. For specific analysis, please ask detailed questions.",
            "confidence": 0.8,
            "success": True
        }

agent = LocalExpertAgent()