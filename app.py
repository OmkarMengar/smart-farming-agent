import os
from typing import Any, Dict

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "smart-farming-agent-secret-key")

AGENT_NAMES = {
    "crop": "Crop Advisory Agent",
    "mandi": "Mandi Price Agent",
    "pest": "Pest & Protection Agent",
}


def get_watsonx_model():
    """Initialize the IBM watsonx.ai Granite model if credentials are present."""
    api_key = os.getenv("IBM_CLOUD_API_KEY", "").strip()
    project_id = os.getenv("WATSONX_PROJECT_ID", "").strip()
    url = os.getenv("WATSONX_URL", "").strip()

    if not api_key or not project_id or not url:
        return None

    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        credentials = Credentials(api_key=api_key, url=url)
        model = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            credentials=credentials,
            project_id=project_id,
            params={"temperature": 0.3, "max_new_tokens": 400},
        )
        return model
    except Exception:
        return None


def fallback_response(agent: str, context: Dict[str, Any], query: str) -> str:
    crop = context.get("crop", "crop")
    location = context.get("location", "your region")
    season = context.get("season", "current season")
    soil = context.get("soil", "well-drained")

    if agent == "crop":
        return (
            f"Crop Advisory Agent: For {crop} in {location}, maintain balanced irrigation and nutrient planning during {season}. "
            f"Check soil moisture, pH, and drainage in {soil} soil before applying fertilizer. "
            f"For your question '{query}', prioritize practical field monitoring and avoid over-irrigation."
        )
    if agent == "mandi":
        return (
            f"Mandi Price Agent: For {crop}, compare nearby mandis in {location}, check quality and moisture grades, and sell when local demand is strong. "
            f"For '{query}', the best strategy is to align harvest timing with market demand and transport costs."
        )
    return (
        f"Pest & Protection Agent: Monitor {crop} for pest pressure during {season}, especially under humid conditions. "
        f"Use integrated pest management, inspect crops weekly, and apply interventions only when threshold levels are reached. "
        f"For '{query}', focus on prevention, monitoring, and safe application timing."
    )


def build_prompt(agent: str, context: Dict[str, Any], query: str) -> str:
    crop = context.get("crop", "crop")
    location = context.get("location", "your region")
    season = context.get("season", "current season")
    soil = context.get("soil", "well-drained")

    if agent == "crop":
        role = "Crop Advisory Agent"
        task = "Provide practical agronomic advice for crop growth, irrigation, nutrient management, and general field health."
    elif agent == "mandi":
        role = "Mandi Price Agent"
        task = "Explain market timing, mandi strategy, grading, and farmer decisions for getting the best price."
    else:
        role = "Pest & Protection Agent"
        task = "Recommend integrated pest management, disease monitoring, and safe crop protection actions."

    return (
        f"You are the {role}. "
        f"Context: crop={crop}, location={location}, season={season}, soil={soil}. "
        f"User question: {query}. "
        f"Task: {task} "
        "Return a concise but useful answer using bullet points and actionable recommendations."
    )


def generate_agent_response(agent: str, context: Dict[str, Any], query: str) -> str:
    model = get_watsonx_model()
    if model is None:
        return fallback_response(agent, context, query)

    prompt = build_prompt(agent, context, query)

    try:
        result = model.generate_text(prompt=prompt)
        if isinstance(result, dict):
            for key in ["results", "generated_text", "output_text", "text"]:
                if key in result:
                    value = result[key]
                    if isinstance(value, list):
                        return str(value[0]) if value and isinstance(value[0], str) else str(value)
                    return str(value)
            return str(result)
        if isinstance(result, list):
            return str(result[0]) if result else "No response generated."
        return str(result)
    except Exception:
        return fallback_response(agent, context, query)


def detect_agent(user_query: str, requested_agent: str) -> str:
    query_lower = (user_query or "").lower()

    if requested_agent in AGENT_NAMES:
        return requested_agent

    if any(keyword in query_lower for keyword in ["market", "mandi", "price", "sell", "sale", "demand", "commission"]):
        return "mandi"

    if any(keyword in query_lower for keyword in ["pest", "disease", "fungus", "insect", "spray", "protect", "control"]):
        return "pest"

    return "crop"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/advice", methods=["POST"])
def advice():
    payload = request.get_json(force=True) or {}
    crop = payload.get("crop", "Wheat").strip() or "Wheat"
    location = payload.get("location", "Punjab").strip() or "Punjab"
    season = payload.get("season", "Rabi").strip() or "Rabi"
    soil = payload.get("soil", "Loamy").strip() or "Loamy"
    agent_name = payload.get("agent", "crop")
    user_query = payload.get("query", "")

    agent = detect_agent(user_query, agent_name)
    context = {
        "crop": crop,
        "location": location,
        "season": season,
        "soil": soil,
    }

    response = generate_agent_response(
        agent,
        context,
        user_query or f"Provide guidance for {crop} in {location}.",
    )

    return jsonify(
        {
            "status": "success",
            "agent": AGENT_NAMES.get(agent, "Crop Advisory Agent"),
            "response": response,
            "crop": crop,
            "location": location,
            "season": season,
            "soil": soil,
        }
    )


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    app.run(host=host, port=port, debug=True)
