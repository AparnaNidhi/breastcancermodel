from tensorflow.keras.models import load_model
model=load_model("breast.h5")

pred=model.predict_classes([[2,2,2,0,1,3,1,2,0]])
print(pred)