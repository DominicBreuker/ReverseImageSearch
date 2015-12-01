import graphlab as gl

images = gl.image_analysis.load_images("./static/shoes/", "auto", with_path = True)

# preprocess images
images['image'] = gl.image_analysis.resize(images['image'], 256, 256, 3)
images['name'] = images['path'].apply(lambda path: path.split('/')[-1])
images.remove_column('path')
images.save('./data/my_shoe_data')

model = gl.load_model("./data/image_net_model")

images['features'] = model.extract_features(images[['image']])
images.save("./data/my_image_data")