import coremltools

CLASS_ID = {0: "air_conditioner",
            1: "car_horn",
            2: "children_playing",
            3: "dog_bark",
            4: "drilling",
            5: "engine_idling",
            6: "gun_shot",
            7: "jackhammer",
            8: "siren",
            9: "street_music"}

output_labels = ["air_conditioner", 
				 "car_horn", 
				 "children_playing",
				 "dog_bark",
				 "drilling",
				 "engine_idling",
				 "gun_shot",
				 "jackhammer",
				 "siren",
				 "street_music"]

scale = 1/255.
coreml_model = coremltools.converters.keras.convert('./output/cnn.h5')
