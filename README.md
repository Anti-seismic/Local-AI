________________________________

		LOCAL AI PROJECT
________________________________

Copyright (c) 2026 Joël KERLÉGUER
Contact: joel.kerleguer@gmail.com

📁 ~/LocalAI/									              ← Project Root
├── run_ai.sh              						      ← Orchestrator
├── vllm_launcher.py       						      ← vLLM Server Launcher
├── upload_server.py       						      ← File Upload Server Launcher (with Safty and Security measures)
├── app.js                 						      ← Frontend Logic ( preserves CORS and marked.parse() )
├── chat.html              					      	← UI, File Picker, Tools Selector
├── style.css              						      ← UI styling
├── tray.py                						      ← System tray (unchanged)
├── debug_ai.sh									            ← Cleans (./debug_ai.sh --kill) and Diagnosis (./debug_ai.sh)
├── run_ai.show									            ← Begins the Show
├── favicon.ico 								            ← Browser tab icon image
├── favicon.svg									            ← Browser tab icon image
├── LocalAI_icon_256.png						        ← Desktop icon image
├── LocalAI_icon_48.png							        ← Desktop icon image
├── LocalAI.desktop								          ← Desktop icon (must be on Desktop too)
📁 ~/LocalAI/models/info/						        ← Repository: VLLM Launching Commands of the Local AI Models in ~/LocalAI/models
├── Qwen3.5-2B-AWQ-4bit.json					      ← AI Models VLLM Launching Commands
📁 ~/LocalAI/models/							          ← Repository: Local AI Models
├── ~/LocalAI/models/Qwen3.5-2B-AWQ-4bit/		← AI Model Qwen3.5-2B-AWQ-4bit
|
📁 ~/LocalAI/virtual_Env/						        ← Repository: Virtual Environments
├── /Qwen3.5-2B-AWQ-4bit/						        ← VLLM Virtual Environment for Qwen3.5-2B-AWQ-4bit Local AI
├── /ProjectUI/									            ← VLLM Virtual Environment for the GUI and the Tools supporting the AI
📁 ~/LocalAI/vllm/								          ← Git clone VLLM
|
📁 ~/LocalAI/data/								          ← Repository: Users Coversations
|
📁 ~/LocalAI/logs/								          ← Repository: Services logs
|
📁 /home/ai-broker/Desktop/
└── LocalAI.desktop								          ← Desktop icon (must be in "~/LocalAI/" and "~/local/share/applications/LocalAI.desktop" too)
