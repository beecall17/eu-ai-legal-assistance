phase 0,
- Configured the .env, and config.settings to have all the api keys and model requried for the entire project in one place (which will be imported when needed whereever)
- added two function which acts as tools (local mcp tools) - 1. to extract article structure, 2. to generate summary 

phase 1. 
- created orchestrator  -- the router logic to route the preferred tool, or return all (both) tools according to query and context.

phase 2.


differnt rag evaluation:
NaiveRAG     | Hit Rate: 75.00% | MRR: 0.513
HybridRAG    | Hit Rate: 84.38% | MRR: 0.597
AdvancedRAG  | Hit Rate: 84.38% | MRR: 0.651

All of the test cases for agentic rag used the advancedrag by the orchestrator. 


📊 FINAL COMPARISON SUMMARY
========================================
• Local vLLM (Llama-3.1-8B-AWQ):
  - Average Latency:  1.74s
  - Throughput Speed: 57.44 tokens/sec
----------------------------------------
• Google Gemini (Gemini 3.6 Flash):
  - Average Latency:  1.80s
  - Throughput Speed: 35.05 tokens/sec

Key Takeaways from Results:
Local vLLM (Llama-3.1-8B-AWQ):

Throughput Speed: 57.44 tokens/sec — Significantly faster raw generation speed.

Reliability: 100% success rate across requests. Zero network overhead or cloud rate limits.

Google Gemini (Gemini 3.6 Flash):

Reliability Issue: Hit a 503 ServiceUnavailableError ("This model is currently experiencing high demand") on 2 out of 3 requests. This is a common bottleneck when hitting free-tier or shared cloud endpoints during high-traffic spikes.

Speed: When it did succeed (Request 2), it finished in 1.80s, which is comparable in raw latency to local inference, but its overall effective throughput suffered due to the dropped requests.


