from weps.neurons.elliott_wave_neuron import ElliottWaveNeuron

import pandas as pd

df = pd.read_csv("~/weps/data/raw_ohlcv/aapl_ohlcv.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

ew_neuron = ElliottWaveNeuron(df)
result = ew_neuron.compute()

print(f"Impulse waves detected: {len(result['impulse_waves'])}")
print(f"Corrective waves detected: {len(result['corrective_waves'])}")
print(f"Wave confidence: {result['wave_confidence']}")
