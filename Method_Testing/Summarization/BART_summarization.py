from transformers import BartForConditionalGeneration, BartTokenizer

# Load BART model and tokenizer
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Sample text
text = """
moreover the central bank initiated measures to prioritise essential imports and restrict capital outflows through appropriate control measures while continuing the requirement for the mandatory sale of foreign exchange to the central bank by licensed banks based on the conversion of repatriated foreign exchange and commenced providing daily guidance on the exchange rate to curtail undue intraday volatilities.moreover the central bank continuous financial sector oversight and adoption of appropriate regulatory measures along with effective communication ensured financial system stability amidst severe socio economic distress.economic price and financial system stability outlook and policieseconomy.the credibility of the central bank is highly related to the independence of the bank.in this regard the envisaged enactment of the new central bank of sri lanka act will contribute immensely to improving the independence and credibility of the central bank which in turn will support the current disinflation episode and further strengthen the anchoring of inflation expectations.globalization and global disinflation federal reserve bank of kansas city conference on monetary policy and uncertainty adapting to changing economy.economic price and financial system stability outlook and policiesyears the surge in imported prices as well as increases in the cost of non food categories such as restaurants and hotels health and education etc.this continued reduction in core inflation was attributed to the strong policy measures taken by the central bank to address the build up of demand driven inflationary pressures and adverse inflation expectations.the government and the central bank initiated measures to limit foreign exchange outflows while taking initiatives to promote inflows. import demand was reduced notably reflecting the impact of significantly tightened monetary policy and subdued demand conditions.the central bank played major role in managing foreign exchange to ensure the supply of essential goods and services under extremely challenging circumstances during.figure balance of paymentssource central bank of sri lankatrade balance current account balance overall balance us billion
"""

# Preprocess text for BART
inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

# Generate summary
summary_ids = model.generate(inputs, num_beams=4, min_length=50, max_length=100, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
