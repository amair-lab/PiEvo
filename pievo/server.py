import threading
import logging
from flask import Flask, jsonify, render_template, send_from_directory
import os
from dataclasses import asdict

# Configure logging to avoid Flask spam
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
pievo_instance = None

def get_pievo():
    return pievo_instance

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metrics')
def get_metrics():
    pievo = get_pievo()
    if not pievo or not pievo.track:
        return jsonify([])
    
    # Convert dataclass objects to dicts
    metrics = [asdict(m) for m in pievo.track.theoretical_metrics_logs]
    return jsonify(metrics)

@app.route('/api/logs/experiment')
def get_experiment_logs():
    pievo = get_pievo()
    if not pievo or not pievo.track:
        return jsonify([])
    
    logs = [asdict(l) for l in pievo.track.experiment_guidance_logs]
    return jsonify(logs)

@app.route('/api/logs/principle')
def get_principle_logs():
    pievo = get_pievo()
    if not pievo or not pievo.track:
        return jsonify([])
    
    logs = [asdict(l) for l in pievo.track.principle_guidance_logs]
    return jsonify(logs)

@app.route('/api/logs/hypothesis')
def get_hypothesis_logs():
    pievo = get_pievo()
    if not pievo or not pievo.track:
        return jsonify([])
    
    logs = [asdict(l) for l in pievo.track.hypothesis_guidance_logs]
    return jsonify(logs)

@app.route('/api/status')
def get_status():
    pievo = get_pievo()
    if not pievo:
        return jsonify({"status": "Not Initialized"})
    return jsonify({"status": getattr(pievo, "status", "Running")})

@app.route('/api/principles')
def get_principles():
    pievo = get_pievo()
    if not pievo:
        return jsonify([])
    
    principles_data = []
    for pid, text in pievo.principles.items():
        belief = pievo.principle_beliefs.get(pid, 0.0)
        rational = pievo.principles_rationals.get(pid, "")
        principles_data.append({
            "id": pid,
            "text": text,
            "belief": belief,
            "rational": rational
        })
    
    # Sort by belief descending
    principles_data.sort(key=lambda x: x["belief"], reverse=True)
    return jsonify(principles_data)

@app.route('/api/history')
def get_history():
    pievo = get_pievo()
    if not pievo:
        return jsonify([])
    
    # pievo.history is a list of tuples (hypothesis, outcome)
    history_data = [{"hypothesis": h, "outcome": o} for h, o in pievo.history]
    return jsonify(history_data)

@app.route('/api/figures/<path:filename>')
def get_figure(filename):
    pievo = get_pievo()
    if not pievo or not pievo.track:
        return "No tracker available", 404
    
    # Serve from the metrics directory
    metrics_dir = os.path.abspath(pievo.track.log_dir)
    return send_from_directory(metrics_dir, filename)

def run_flask(port):
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)

def start_server(pievo, port=8085):
    global pievo_instance
    pievo_instance = pievo
    
    # Ensure templates directory exists
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
        
    server_thread = threading.Thread(target=run_flask, args=(port,), daemon=True)
    server_thread.start()
    print(f"📊 Visualization server started at http://localhost:{port}")
