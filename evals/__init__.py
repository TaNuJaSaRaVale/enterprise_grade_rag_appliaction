"""
Evaluation package for the Enterprise RAG system.

Run modules directly, e.g.:
    python -m evals.pipeline --limit 3

Intentionally no eager imports here: `python -m evals.<module>` imports this
package first, and importing submodules here would load them twice (RuntimeWarning)
and pull in the whole RAG stack for any import from `evals`.
"""
