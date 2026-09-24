"""The DeepEcho API.

One FastAPI application over two engines that were built separately:

    app/            the Supabase-backed shell: uploads, persistence, history
    ../chat.py      the grounded assistant, over the corpus in kb/
    ../detect.py    marine.pt, in a subprocess

The shell owns storage, deployment and the route surface. The engines own
inference and grounding. Nothing in the shell reaches into rag.py directly.
"""
