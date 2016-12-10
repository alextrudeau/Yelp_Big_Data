import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

states = ["PA", "NC", "NV", "WI", "AZ", "IL"]
first = True
Categories = {}
CategoryL = []

with open('Business_Categories.csv', 'rU') as csvfile:
	datareader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in datareader:
		if first:
			for category in row:
				CategoryL += [category]
			first = False
		else:
			counter = 0
			for subcategory in row:
				if subcategory != "":
					Categories[subcategory] = CategoryL[counter]
				counter += 1

# categories -> zip codes -> business id ->
businesses = {}

# business id -> category, zip
business_search = {}

business_file = open('yelp_academic_dataset_business.json', 'r')
for line in business_file:
	business_data = json.loads(line)
	business_id = str(business_data["business_id"])
	business_address = business_data["full_address"].encode('utf-8')

	address_split = business_address.split()
	if len(address_split) > 0 and len(address_split[-1]) == 3:
		business_zip = None
		business_state = None
	elif len(address_split) >= 2:
		business_zip = str(address_split[-1])
		business_state = str(address_split[-2])
	else:
		business_zip = None
		business_state = None

	if business_state != None and business_state in states and business_data["categories"] != [] and str(business_data["categories"][0]) in Categories:
		business_type = str(business_data["categories"][0]) # taking first category - might change later!
		global_category = Categories[business_type]
		if global_category not in businesses:
			businesses[global_category] = {}
		if business_zip not in businesses[global_category]:
			businesses[global_category][business_zip] = {}
		businesses[global_category][business_zip][business_id] = {}
		businesses[global_category][business_zip][business_id]["reviews"] = []
		businesses[global_category][business_zip][business_id]["population"] = 0
		businesses[global_category][business_zip][business_id]["income"] = 0
		businesses[global_category][business_zip][business_id]["poverty"] = 0.0
		businesses[global_category][business_zip][business_id]["num reviews"] = 0
		business_search[business_id] = {}
		business_search[business_id]["category"] = global_category
		business_search[business_id]["zip code"] = business_zip

review_file = open('yelp_academic_dataset_review.json', 'r')
for line in review_file:
	review_data = json.loads(line)
	business_id = str(review_data["business_id"])
	rating = review_data["stars"]
	if business_id in business_search:
		zip_code = business_search[business_id]["zip code"]
		category = business_search[business_id]["category"]
		businesses[category][zip_code][business_id]["reviews"] += [rating]
		businesses[category][zip_code][business_id]["num reviews"] += 1

with open('ZipPopIncPovNet.csv', 'rU') as csvfile:
	datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in datareader:
		line = row[0]
		line = line.split(",")
		code = line[0]
		population = line[1]
		poverty = line[3]
		income = line[2]
		if code != "Zip_Code" and income != "Income":
			income = int(income)
			poverty = float(poverty)
			population = int(population)
			for category in businesses:
				if code in businesses[category]:
					for businessID in businesses[category][code]:
						businesses[category][code][businessID]["population"] = population
						businesses[category][code][businessID]["poverty"] = poverty
						businesses[category][code][businessID]["income"] = income

avg_rating = []
income_data = []
num_reviews_data = []
avg_dic = {}

busCat = "Travel & Transportation"

for zipCode in businesses[busCat]:
	for establishment in businesses[busCat][zipCode]:
		if float(businesses[busCat][zipCode][establishment]["num reviews"]) != 0.0 and businesses[busCat][zipCode][establishment]["income"] != 0:
			num_reviews_data += [businesses[busCat][zipCode][establishment]["num reviews"]]
			avg_rating += [float(sum(businesses[busCat][zipCode][establishment]["reviews"])) / float(businesses[busCat][zipCode][establishment]["num reviews"])]



for i in range(len(num_reviews_data)):
	if num_reviews_data[i] not in avg_dic:
		avg_dic[num_reviews_data[i]] = []
	avg_dic[num_reviews_data[i]] += [avg_rating[i]]

num_reviews_arr = avg_dic.keys()
rating_arr = []


for key in num_reviews_arr:
	rating_arr += [float(sum(avg_dic[key])) / float(len(avg_dic[key]))]
"""

for zipCode in businesses[busCat]:
	for establishment in businesses[busCat][zipCode]:
		if businesses[busCat][zipCode][establishment]["income"] != 0:
			income_data += [businesses[busCat][zipCode][establishment]["income"]]
			num_reviews_data += [businesses[busCat][zipCode][establishment]["num reviews"]]


for j in range(len(income_data)):
	if income_data[j] not in avg_dic:
		avg_dic[income_data[j]] = []
	avg_dic[income_data[j]] += [num_reviews_data[j]]

income_arr = avg_dic.keys()
num_reviews_arr = []

for key in income_arr:
	num_reviews_arr += [float(sum(avg_dic[key])) / float(len(avg_dic[key]))]


for zipCode in businesses[busCat]:
	for establishment in businesses[busCat][zipCode]:
		if float(businesses[busCat][zipCode][establishment]["num reviews"]) != 0.0 and businesses[busCat][zipCode][establishment]["income"] != 0:
			income_data += [businesses[busCat][zipCode][establishment]["income"]]
			avg_rating += [float(sum(businesses[busCat][zipCode][establishment]["reviews"])) / float(businesses[busCat][zipCode][establishment]["num reviews"])]



for i in range(len(income_data)):
	if income_data[i] not in avg_dic:
		avg_dic[income_data[i]] = []
	avg_dic[income_data[i]] += [avg_rating[i]]

income_arr = avg_dic.keys()
rating_arr = []


for key in income_arr:
	rating_arr += [float(sum(avg_dic[key])) / float(len(avg_dic[key]))]
"""

X = []

for inc in num_reviews_arr:
	X += [[1, inc]]

print "X Len: ", len(X)

Y = np.matrix(rating_arr).T

print "Y Len: ", len(Y)

XReal = np.matrix(X)
XT = XReal.T


XTX = XT.dot(XReal)

XTY = XT.dot(Y)

invXTX = XTX.I

Theta = invXTX.dot(XTY)

ThetaL = Theta.tolist()

b = ThetaL[0][0]
m = ThetaL[1][0]

print m
print b

fitX = list(set(num_reviews_arr))
fitY = []

for element in fitX:
	fitY += [element*m + b]

plt.plot(num_reviews_arr, rating_arr, 'ro', fitX, fitY, 'p-')
axes = plt.gca()
axes.set_xlim([0,max(num_reviews_arr) + 10])
axes.set_ylim([0,5.5])
plt.show()








		