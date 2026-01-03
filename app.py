import os
from flask import Flask
from flask_cors import CORS

import config
from init_app import init_available_collections
from routes import api as api_blueprint
from routes_user import user_bp


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    init_available_collections()

    app.register_blueprint(api_blueprint)
    app.register_blueprint(user_bp)
    return app


if __name__ == "__main__":
    app = create_app()
    print("Starting app with collections:", {
        "COL_S2": config.COL_S2,
       
    })
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
