import gradio as gr
import requests
import json
import sys


API_URL = "http://localhost:8000"

def query_agent(user_query):
    try:
        response = requests.post(
            f"{API_URL}/api/agent/query",
            json={"query": user_query},
            timeout=30
        )
        
        print(f"DEBUG: Status Code: {response.status_code}") 
        if response.status_code == 200:
            result = response.json()
            print(f"DEBUG: Response keys: {list(result.keys())}") 
            
            formatted = f"**Query:** {user_query}\n\n"
            
            if result.get('tool_used'):
                formatted += f"**Tool Used:** {result['tool_used']}\n\n"
              
                tool_result = json.dumps(result.get('tool_result', {}), indent=2)
                formatted += f"<details><summary> Tool Result</summary>\n```json\n{tool_result}\n```\n</details>\n\n"
            
          
            if 'answer' in result:
                formatted += f"**Answer:**\n{result['answer']}"
            else:
                formatted += f"**Response:**\n{json.dumps(result, indent=2)}"
            
            return formatted
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Connection error: {str(e)}"


iface = gr.Interface(
    fn=query_agent,
    inputs=gr.Textbox(
        label="Ask a question",
        placeholder="E.g., 'Calculate ROI for a $300k property with $2000 monthly rent' or 'Is $450,000 a good price?'",
        lines=3
    ),
    outputs=gr.Markdown(label="Agent Response"),
    title=" GeoInsight AI - Local Expert Agent",
    description="Ask questions about real estate, investments, or neighborhoods. The AI agent will analyze and provide insights.",
    examples=[
        "Calculate ROI for $300k property with $2000 rent",
        "Is $450,000 a good price for a house?",
        "What can I get for $1500 monthly rent?",
        "Investment analysis: $500k house, $3000 rent, 20% down",
        "Price check: $750,000 property"
    ]
)

if __name__ == "__main__":
   
    print(f" Local URL: http://localhost:7861")
   
  
    ports_to_try = [7861, 7862, 8888, 8080]
    
    for port in ports_to_try:
        try:
            print(f"\nTrying port {port}...")
            iface.launch(
                server_name="127.0.0.1",  
                server_port=port,
                share=False,
                debug=False
            )
            break  
        except Exception as e:
            print(f" Port {port} failed: {str(e)[:100]}")
            if port == ports_to_try[-1]:
                print(" All ports failed. Trying with share=True...")
                iface.launch(share=True)  