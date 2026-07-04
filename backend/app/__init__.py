"""YouTube Success Predictor — backend application package.

The pipeline, end to end:

    YouTube URL / local file
        -> app.dataset      download + metadata + media extraction
        -> app.tribe        official TRIBE v2 inference -> brain activity (T x V)
        -> app.models       BrainEncoder (T x V -> 512-d) + multitask head
        -> app.training      supervised training against observed video success
        -> app.services      inference orchestration, caching, persistence
        -> app.api           FastAPI surface consumed by the React dashboard

Every subpackage is import-safe without a GPU or the `tribev2` weights present;
heavy resources are acquired lazily so that tooling, tests and the API can be
imported in any environment.
"""

__version__ = "0.1.0"
