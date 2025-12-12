import os
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

"""Flask HTTP Server for Hint Service"""
from flask import Flask, request, jsonify
import os
import sys

sys.path.insert(0, '/workspace/hint_service')
from hint_core import generate_hint

app = Flask(__name__)

@app.route('/hint', methods=['POST'])
def hint_endpoint():
    try:
        data = request.json
        if not data.get('problem_id'):
            return jsonify({'success': False, 'error': 'problem_id required'}), 400
        
        print(f"[Hint] problem_id={data['problem_id']}, preset={data.get('preset')}")
        
        result = generate_hint(
            problem_id=data['problem_id'],
            user_code=data.get('user_code', ''),
            star_count=data.get('star_count', 0),
            preset=data.get('preset', '초급'),
            custom_components=data.get('custom_components', {}),
            previous_hints=data.get('previous_hints', []),
            problem_data=data.get('problem_data', {}),
            last_hint_info=data.get('last_hint_info', {})
        )
        
        print(f"[Hint] Success")
        return jsonify({'success': True, 'data': result})
    
    except Exception as e:
        print(f"[Hint Error] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        'status': 'alive',
        'message': 'Hint service running',
        'openai_configured': bool(os.environ.get('OPENAI_API_KEY'))
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'hint_service': 'healthy',
        'openai_key': 'configured' if os.environ.get('OPENAI_API_KEY') else 'missing'
    })

if __name__ == '__main__':
    port = 8080
    print(f"Starting server on port {port}...")
    print(f"OpenAI API Key: {'configured' if os.environ.get('OPENAI_API_KEY') else 'MISSING'}")
    app.run(host='0.0.0.0', port=port, debug=False)