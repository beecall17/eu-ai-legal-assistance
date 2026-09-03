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

