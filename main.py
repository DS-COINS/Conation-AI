from flask import Flask, jsonify, request
from flask_restx import Resource, Api
from conation_recomender import okt_exactR
from conation_recomender_W2V import word2vec_similarR

app = Flask(__name__)
api = Api(app)
app.config['JSON_AS_ASCII'] = False

@api.route('/')
class Home(Resource):
    def get(self):
        return "hello world"

@api.route('/exactRecommend')
class ERecommend(Resource):

    def post(self):

        title = request.json.get('title')
        category = request.json.get('category')

        jsonData = okt_exactR(title, category)

        return jsonify({
            'result': jsonData
        })

@api.route('/similarRecommed')
class SRecommend(Resource):
    def post(self):
        title2 = request.json.get('title')
        category2 = request.json.get('category')

        jsonData2 = word2vec_similarR(title2, category2)

        return jsonify({
            'result': jsonData2
        })

if __name__ == '__main__':
    app.run(debug=True)