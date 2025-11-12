# neuro_fuzzy
A transformer AI model for early fault detection




python predict.py --method duval --file test_model.csv
If that works, then:

bash
Copy code
python predict.py --method all --file test_model.csv
will confirm your unified setup is intact.










python predict.py --method rogers --file test_model.csv
python predict.py --method drm --file test_model.csv
✅ This will only predict using the specified method’s trained model(s).

⚙️ 3️⃣ Single Sample Prediction (Manual Input)
If you just want to test one gas reading directly (without any file):

bash
Copy code
python predict.py --method duval
or all methods together:

bash
Copy code
python predict.py --method all
✅ This will use the built-in example in your script:

python
Copy code
gas_sample = {"CH4": 500, "C2H4": 500, "C2H2": 50, "H2": 10, "C2H6": 5, "CO": 0}