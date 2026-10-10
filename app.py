"""
app.py - Flask web server for Smartphone Addiction Prediction
"""
import os, sys, json, pickle, tempfile, shutil
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template

sys.path.insert(0, str(Path(__file__).parent))
from utils.extractor import extract_user_data

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

MODELS_DIR = Path('models')

def load_models():
    with open(MODELS_DIR/'random_forest.pkl','rb') as f: rf = pickle.load(f)
    with open(MODELS_DIR/'decision_tree.pkl','rb') as f: dt = pickle.load(f)
    with open(MODELS_DIR/'label_encoder.pkl','rb') as f: le = pickle.load(f)
    with open(MODELS_DIR/'feature_cols.json') as f: fc = json.load(f)
    with open(MODELS_DIR/'accuracy_summary.json') as f: acc = json.load(f)
    return rf, dt, le, fc, acc

try:
    RF, DT, LE, FEATURE_COLS, ACC_SUMMARY = load_models()
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    print(f"Models not loaded: {e}. Run train_model.py first.")

def generate_explanation(features, label, rf_proba):
    total=features.get('total_screen_time_min',0); social=features.get('social_time_min',0)
    prod=features.get('productivity_time_min',0); video=features.get('video_time_min',0)
    gaming=features.get('gaming_time_min',0); night=features.get('night_10pm_2am_min',0)
    npct=features.get('night_usage_pct',0); sprod=features.get('social_prod_ratio',0)
    notif_hr=features.get('notification_per_hour',0); unl_hr=features.get('unlock_frequency_per_hour',0)
    lines=[]
    lines.append(f"📱 Total screen time is {total} minutes/day ({total//60}h {total%60}m).")
    if total>360: lines.append("⚠ Critically high (6+ hours) — well above the healthy 2-hour daily limit.")
    elif total>240: lines.append("⚠ Exceeds the recommended 4-hour daily limit for adults.")
    else: lines.append("✅ Total screen time is within an acceptable range.")
    if social>180: lines.append(f"📲 Social media dominates at {social} min/day — key driver of addictive patterns.")
    elif social>120: lines.append(f"📲 Social media usage ({social} min) is elevated and contributes to the prediction.")
    else: lines.append(f"📲 Social media time ({social} min) appears balanced.")
    if sprod>3: lines.append(f"⚖ Social-to-Productivity ratio is {sprod:.1f}x — social apps overwhelm productive use.")
    elif sprod>2: lines.append(f"⚖ Social-to-Productivity ratio is {sprod:.1f}x — social use notably exceeds productive use.")
    else: lines.append(f"⚖ Social-to-Productivity ratio ({sprod:.1f}x) is relatively balanced.")
    if npct>15: lines.append(f"🌙 Night usage ({npct:.1f}% of daily total) is very high — disrupts sleep cycles.")
    elif npct>10: lines.append(f"🌙 Significant late-night usage ({npct:.1f}%) detected — key addiction indicator.")
    else: lines.append(f"🌙 Night-time usage ({npct:.1f}%) is within acceptable levels.")
    if notif_hr>8: lines.append(f"🔔 {notif_hr:.1f} notifications/hour — constant interruptions reinforce compulsive checking.")
    elif notif_hr>5: lines.append(f"🔔 Notification rate ({notif_hr:.1f}/hour) is high and triggers frequent pickups.")
    if unl_hr>8: lines.append(f"🔓 {unl_hr:.1f} unlocks/hour suggests compulsive checking behaviour.")
    elif unl_hr>5: lines.append(f"🔓 Unlock frequency ({unl_hr:.1f}/hour) is elevated.")
    if gaming>60: lines.append(f"🎮 Gaming time ({gaming} min) contributes to total screen exposure.")
    if video>90: lines.append(f"📺 Video streaming ({video} min) forms a significant portion of screen time.")
    proba_max=max(rf_proba)
    lines.append(f"\n🤖 Model confidence: {proba_max*100:.1f}% ({label} risk).")
    lines.append(f"   RF probabilities — Low: {rf_proba[0]*100:.1f}%, Moderate: {rf_proba[1]*100:.1f}%, High: {rf_proba[2]*100:.1f}%")
    return '\n'.join(lines)

def generate_recommendations(features, label):
    recs=[]
    total=features.get('total_screen_time_min',0); social=features.get('social_time_min',0)
    night=features.get('night_10pm_2am_min',0); npct=features.get('night_usage_pct',0)
    sprod=features.get('social_prod_ratio',0); unl_hr=features.get('unlock_frequency_per_hour',0)
    nf_hr=features.get('notification_per_hour',0); gaming=features.get('gaming_time_min',0)
    video=features.get('video_time_min',0)
    if total>360: recs.append("🚨 Set a strict 3-hour daily screen time limit using Digital Wellbeing / Screen Time.")
    elif total>240: recs.append("⏱ Aim to reduce total daily screen time to under 3 hours with built-in app timers.")
    if social>180:
        recs.append("📵 Enforce 30-min daily social media limits through app timers.")
        recs.append("🗑 Uninstall or grey-scale the most-used social apps to reduce visual appeal.")
    elif social>120: recs.append("📲 Set a 1-hour social media budget per day and track it weekly.")
    if sprod>2: recs.append("💼 Schedule focused 'productivity blocks' (90 min) with social apps blocked via Focus Mode.")
    if npct>10 or night>20:
        recs.append("🌙 Enable Bedtime Mode from 9:30 PM — phone goes grayscale and silent.")
        recs.append("🛏 Charge your phone outside the bedroom to prevent late-night usage.")
    if unl_hr>5:
        recs.append("🔒 Turn off non-essential notifications to reduce compulsive checking impulses.")
        recs.append("⏰ Practice the '20-minute rule' — check your phone every 20 minutes only.")
    if nf_hr>5: recs.append("🔕 Batch notifications: check messages only 3 times per day.")
    if gaming>60: recs.append("🎮 Set a 30-minute daily gaming limit and play only after completing tasks.")
    if video>90: recs.append("📺 Replace passive scrolling with intentional viewing — pre-pick content.")
    recs.append("🚶 Replace one phone session per day with a 10-minute walk or stretching.")
    recs.append("📔 Keep a weekly phone journal to track how usage makes you feel.")
    if label=='Low': recs.insert(0,"✅ Great job! Your phone habits are largely balanced. Minor tweaks below:")
    return recs[:8]

def extract_features_from_uploads(files):
    tmp = tempfile.mkdtemp()
    try:
        for f in files:
            filename = f.filename.replace('/','_').replace('\\','_')
            if filename: f.save(os.path.join(tmp, filename))
        return extract_user_data(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/model-info')
def model_info():
    if not MODELS_LOADED: return jsonify({'error':'Models not trained'}),503
    return jsonify(ACC_SUMMARY)

@app.route('/api/predict', methods=['POST'])
def predict():
    if not MODELS_LOADED: return jsonify({'error':'Run train_model.py first.'}),503
    age=request.form.get('age','Unknown'); gender=request.form.get('gender','Unknown')
    files=request.files.getlist('screenshots')
    if not files or all(f.filename=='' for f in files):
        return jsonify({'error':'No screenshots uploaded.'}),400
    try:
        features = extract_features_from_uploads(files)
    except Exception as e:
        return jsonify({'error':f'Extraction failed: {e}'}),500
    feat_vec = np.array([[features.get(c,0) for c in FEATURE_COLS]])
    rf_pred_enc=RF.predict(feat_vec)[0]; rf_proba=RF.predict_proba(feat_vec)[0].tolist()
    dt_pred_enc=DT.predict(feat_vec)[0]; dt_proba=DT.predict_proba(feat_vec)[0].tolist()
    rf_label=LE.inverse_transform([rf_pred_enc])[0]
    dt_label=LE.inverse_transform([dt_pred_enc])[0]
    final_label=rf_label
    explanation=generate_explanation(features,final_label,rf_proba)
    recommendations=generate_recommendations(features,final_label)
    fi_global=dict(zip(FEATURE_COLS, RF.feature_importances_.tolist()))
    return jsonify({
        'age':age,'gender':gender,
        'prediction':{'label':final_label,'rf_label':rf_label,'dt_label':dt_label,
                      'rf_proba':rf_proba,'dt_proba':dt_proba,'label_order':['Low','Moderate','High']},
        'features':features,'feature_cols':FEATURE_COLS,'feature_importance':fi_global,
        'explanation':explanation,'recommendations':recommendations,
        'model_accuracy':{'rf':ACC_SUMMARY.get('rf_test_acc'),'dt':ACC_SUMMARY.get('dt_test_acc')}
    })

if __name__=='__main__':
    app.run(debug=True,port=5000)