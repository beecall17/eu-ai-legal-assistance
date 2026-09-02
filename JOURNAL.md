phase 0,
- Configured the .env, and config.settings to have all the api keys and model requried for the entire project in one place (which will be imported when needed whereever)
- added two function which acts as tools (local mcp tools) - 1. to extract article structure, 2. to generate summary 

phase 1. 
- created orchestrator  -- the router logic to route the preferred tool, or return all (both) tools according to query and context.

phase 2.
Since this is the legal document,
parsed with hybird markdownsplitter for parents to have entire article/chapter/annex context and recursive splitter for child to have high quality chuks of child with parents metadata attached. 
 -- High quality chuks/data than random/basic splitting on all. 