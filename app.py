import graphlab as gl
from flask import Flask, flash, redirect, render_template, request, url_for

def load_image(url):
    return gl.image_analysis.resize(gl.Image(url), 256,256,3)

def find_similar_dresses(query_image, images, deep_model, nn_model):
    sf = graphlab.SFrame()
    sf['image'] = [query_image]
    sf['features'] = deep_model.extract_features(sf)
    query_result = nn_model.query(sf, k=9, verbose=True)
    return images.join(query_result, on = {'id':'reference_label'}).sort('distance')

app = Flask(__name__)
app.secret_key = 'super_secret_key'

@app.route('/', methods = ["GET", "POST"])
def search_form():
    if request.method == "GET":
        return render_template('search_form.html', result = None)
    else:
        url = request.form['url']
        query_image = load_image(url)
        similar_images = find_similar_dresses(query_image, images, deep_model, nn_model)
        return render_template('search_form.html', result = similar_images, id = 'results')

@app.route('/search', methods = ["POST"])
def search():
    url = request.form['url']
    # http://productshots1.modcloth.net/productshots/0122/9494/1303e63f8fc4c4a592c960d0f03115c5.jpg
    query_image = load_image(url)
    similar_images = find_similar_dresses(query_image, images, deep_model, nn_model)
    return render_template('search_results.html', result = similar_images)

if __name__ == '__main__':
    app.debug = True
    images = gl.SFrame("./data/my_image_data")
    images = images.add_row_number('id')
    deep_model = gl.load_model("./data/image_net_model")
    nn_model = gl.nearest_neighbors.create(images, features=['features'])
    app.run()