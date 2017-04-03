import json
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from scipy.stats.stats import pearsonr
reviewData = open("/Users/etheredgej/Desktop/yelp_dataset_challenge_round9/yelp_academic_dataset_review.json", "r").readlines()

businessInfo = {}    # Make an empty dictionary
for d in reviewData[0:5000]:
  parsed = json.loads(d)
  business = parsed["business_id"]
  score = parsed["stars"]
  text = parsed["text"]
  tb = TextBlob(text)
  polarity = tb.sentiment.polarity
  businessData = businessInfo.get(business, [])  # Get the list of scores and polarities out of businessInfo. If there's no entry for that business yet, get an empty list.
  businessData.append((score, polarity,))         # Append a tuple to businessData. businessData is always a list of tuples where the first value is the score and the second value is the polarity.
  businessInfo[business] = businessData

for key, value in businessInfo.items():
    print (key, value)

businessSlopes = {}      # Make an empty dictionary that will be a business (key) vs slopes (value) dictionary
businessCorrelations = {}      # Make an empty dictionary that will be a business (key) vs correlation (value) dictionary
scores = []
polarities = []
for business, scoreAndPolarity in businessInfo.iteritems():      # Iterates over all the items in the dictionary using the form "key, value".
  for x in range(len(scoreAndPolarity)): # creates a separate list for each business of the separate scores and polarities
  	scores.append((scoreAndPolarity[x])[0])
  	polarities.append((scoreAndPolarity[x])[1])
  correlation = pearsonr(scores, polarities)
  businessCorrelations[business] = correlation
  x=np.array(polarities)
  y=np.array(scores)
  m, b = np.polyfit(x, y, 1)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(x, y, 'bo')
  ax.plot(x, m*x + b, 'r-')
  ax.set_title(business) 
  ax.annotate("Slope="+str(round(m,4)), xy=(1, 0.1), xycoords='axes fraction', fontsize=16, horizontalalignment='right', verticalalignment='center')
  ax.annotate("Pearson correlation="+str(round(correlation[0],4)), xy=(1, 0.05), xycoords='axes fraction', fontsize=16, horizontalalignment='right', verticalalignment='center')
  ax.set_xlabel('text sentiment polarity')
  ax.set_ylabel('star rating')
  ax.set_ylim([0,5])
  ax.set_xlim([-1,1])
#  plt.show()
  filename = "%s.png" % business
  fig.savefig(filename)
  plt.close()
  businessSlopes[business] = m

# show business with associated slopes
for business, slopes in businessSlopes.iteritems():
	print (business, slopes) 

# show business with associated correlation
for business, correlations in businessCorrelations.iteritems():
  print (business, correlations) 
