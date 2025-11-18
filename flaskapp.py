from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "b81f5eafd1e94d5eadb1afc0f23028e98374264ff19c"

# Load model (make sure model.pkl exists in same folder)
MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Place your trained model file named '{MODEL_PATH}' in the project folder.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# Questions in same order as your Streamlit chatbot
QUESTIONS = [
    "What is your gender? (Male/Female)",
    "Are you married? (Yes/No)",
    "How many dependents do you have? (0/1/2/3+)",
    "What is your education level? (Graduate/Not Graduate)",
    "Are you self-employed? (Yes/No)",
    "What is your monthly applicant income?",
    "What is your monthly co-applicant income?",
    "What is the loan amount you are requesting?",
    "What is the loan term in days?",
    "What is your credit history score? (300-850)",
    "What is your area? (Urban/Semiurban/Rural)"
]


def preprocess_data(gender, married, dependents, education, employed, credit, area,
                    ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term):
    """
    Exact preprocessing from your Streamlit code.
    Returns a list of 14 features in the same order used previously.
    """
    try:
        male = 1 if str(gender).lower() == "male" else 0
        married_yes = 1 if str(married).lower() == "yes" else 0

        dep = str(dependents)
        if dep == '1':
            dependents_1, dependents_2, dependents_3 = 1, 0, 0
        elif dep == '2':
            dependents_1, dependents_2, dependents_3 = 0, 1, 0
        elif dep == "3+":
            dependents_1, dependents_2, dependents_3 = 0, 0, 1
        else:
            dependents_1, dependents_2, dependents_3 = 0, 0, 0

        not_graduate = 1 if str(education).lower() == "not graduate" else 0
        employed_yes = 1 if str(employed).lower() == "yes" else 0
        semiurban = 1 if str(area).lower() == "semiurban" else 0
        urban = 1 if str(area).lower() == "urban" else 0

        ApplicantIncomelog = np.log(float(ApplicantIncome))
        totalincomelog = np.log(float(ApplicantIncome) + float(CoapplicantIncome))
        LoanAmountlog = np.log(float(LoanAmount))
        Loan_Amount_Termlog = np.log(float(Loan_Amount_Term))

        # EXACT credit rule from your Streamlit: >=800 and <=1000 => 1 else 0
        credit_flag = 1 if (float(credit) >= 800 and float(credit) <= 1000) else 0

        return [
            credit_flag, ApplicantIncomelog, LoanAmountlog, Loan_Amount_Termlog, totalincomelog,
            male, married_yes, dependents_1, dependents_2, dependents_3,
            not_graduate, employed_yes, semiurban, urban
        ]
    except Exception as e:
        # Return None to indicate an error in preprocessing
        return None


# ----------------- Routes -----------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


# Prediction page (manual form)
@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    result = None
    details = None
    if request.method == "POST":
        try:
            gender = request.form.get("gender")
            married = request.form.get("married")
            dependents = request.form.get("dependents")
            education = request.form.get("education")
            employed = request.form.get("employed")
            credit = request.form.get("credit")
            area = request.form.get("area")
            ApplicantIncome = request.form.get("ApplicantIncome")
            CoapplicantIncome = request.form.get("CoapplicantIncome")
            LoanAmount = request.form.get("LoanAmount")
            Loan_Amount_Term = request.form.get("Loan_Amount_Term")

            # fallback defaults & conversions
            ApplicantIncome = float(ApplicantIncome) if ApplicantIncome not in (None, "") else 0.0
            CoapplicantIncome = float(CoapplicantIncome) if CoapplicantIncome not in (None, "") else 0.0
            LoanAmount = float(LoanAmount) if LoanAmount not in (None, "") else 1.0
            Loan_Amount_Term = float(Loan_Amount_Term) if Loan_Amount_Term not in (None, "") else 360.0
            credit_val = float(credit) if credit not in (None, "") else 0.0

            features = preprocess_data(gender, married, dependents, education, employed, credit_val, area,
                                       ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term)
            if features is None:
                raise ValueError("Invalid inputs for preprocessing.")

            pred_raw = model.predict([features])[0]

            # the Streamlit code sometimes expects 'Y'/'N' else 1/0, handle both
            approved = False
            if str(pred_raw) == "Y" or str(pred_raw).upper() == "Y" or str(pred_raw) == "1" or str(pred_raw) == "True":
                approved = True

            if approved:
                result = "✅ Loan Status: Approved 🎉"
            else:
                result = "☠️ Loan Status: Rejected ❌"

            details = {
                "gender": gender, "married": married, "dependents": dependents, "education": education,
                "employed": employed, "ApplicantIncome": ApplicantIncome, "CoapplicantIncome": CoapplicantIncome,
                "LoanAmount": LoanAmount, "Loan_Amount_Term": Loan_Amount_Term, "credit": credit_val, "area": area
            }

        except Exception as e:
            result = f"Error during prediction: {str(e)}"

    return render_template("prediction.html", result=result, details=details)


# Chatbot endpoints: serve page + API for chat steps
@app.route("/chatbot")
def chatbot_page():
    # initialize session for chat
    session["current_step"] = 0
    session["responses"] = {}
    first_question = QUESTIONS[0]
    return render_template("chatbot.html", first_question=first_question)


@app.route("/chatapi", methods=["POST"])
def chatapi():
    """
    AJAX endpoint that receives a single answer, validates it (per-step),
    stores it in session, and returns the next question or final prediction.
    """
    data = request.form or request.get_json() or {}
    user_answer = (data.get("message") or "").strip()

    current_step = int(session.get("current_step", 0))
    responses = session.get("responses", {})

    # Save answer for current step
    responses[str(current_step)] = user_answer
    session["responses"] = responses

    # Validation: numeric steps 5,6,7,8 in your Streamlit code; step 9 is credit validation
    if current_step in [5, 6, 7, 8]:
        try:
            float(user_answer)
        except ValueError:
            return jsonify({"reply": "Please enter a valid number.", "end": False})
    if current_step == 9:
        # credit score validation
        try:
            score = float(user_answer)
            if not (0 <= score <= 1000):
                return jsonify({"reply": "Credit score must be between 0 and 1000.", "end": False})
        except ValueError:
            return jsonify({"reply": "Please enter a valid credit score.", "end": False})

    # move to next
    current_step += 1
    session["current_step"] = current_step

    if current_step < len(QUESTIONS):
        next_q = QUESTIONS[current_step]
        return jsonify({"reply": next_q, "end": False})

    # all answered -> prepare features and predict
    try:
        r = session.get("responses", {})
        gender = r.get("0", "")
        married = r.get("1", "")
        dependents = r.get("2", "")
        education = r.get("3", "")
        self_employed = r.get("4", "")
        applicant_income = float(r.get("5", "0") or 0)
        coapplicant_income = float(r.get("6", "0") or 0)
        loan_amount = float(r.get("7", "1") or 1)
        loan_amount_term = float(r.get("8", "360") or 360)
        credit_history = float(r.get("9", "0") or 0)
        property_area = r.get("10", "")

        features = preprocess_data(gender, married, dependents, education, self_employed, credit_history, property_area,
                                   applicant_income, coapplicant_income, loan_amount, loan_amount_term)
        if features is None:
            return jsonify({"reply": "Error in preprocessing provided values.", "end": True})

        pred_raw = model.predict([features])[0]
        approved = False
        if str(pred_raw) == "Y" or str(pred_raw).upper() == "Y" or str(pred_raw) == "1" or str(pred_raw) == "True":
            approved = True

        if approved:
            reply = (
                "✅ Eligible for loan. 🎉\n\n"
                "Next steps:\n"
                "1. Prepare ID & income documents (Aadhaar, PAN, salary slips/bank statements).\n"
                "2. Contact lender/branch or apply online with these documents.\n"
                "3. Loan review & KYC — expect 7–15 business days (may vary).\n"
                "Tips: Maintain timely payments and keep debt low."
            )
        else:
            reply = (
                "❌ Not eligible for loan.\n\n"
                "Suggestions:\n"
                "- Improve credit score (pay bills/loans on time).\n"
                "- Increase applicant income or reduce requested loan amount.\n"
                "- Reapply after improving finances or consider a co-applicant."
            )

        # clear session for next conversation
        session.pop("current_step", None)
        session.pop("responses", None)

        return jsonify({"reply": reply, "end": True, "approved": approved})
    except Exception as e:
        return jsonify({"reply": f"Error during final prediction: {str(e)}", "end": True})


# Run app
if __name__ == "__main__":
    app.run(debug=True)
