import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

"""
Each business will be a row in the array X

"""

def covariance(X,Y,muX,muY):
	runSum = 0
	for i in range(len(X)):
		runSum += (X[i] - muX)*(Y[i] - muY)
	return float(runSum) / float(len(X)-1)

def deviation(X, muX):
	runSum = 0
	for i in range(len(X)):
		runSum += (X[i] - muX)**2
	radical = float(runSum) / float(len(X)-1)
	return radical**(0.5)

def correlation(X,Y):
	sumX = sum(X)
	sumY = sum(Y)
	muX = float(sumX) / float(len(X))
	muY = float(sumY) / float(len(Y))
	cov = covariance(X,Y,muX,muY)
	devX = deviation(X,muX)
	devY = deviation(Y,muY)
	return cov / (devX * devY)

# w = theta - the weight vector
def objective(X, y, w, reg=1e-6):
	err = X.dot(w) - y
	err = float(err.T.dot(err))
	return (err + reg * np.abs(w).sum())/len(y)

def grad_objective(X, y, w):
	return X.T.dot(X.dot(w) - y) / len(y)

def prox(x, gamma):
	for i in range(len(x)):
		if np.abs(x[i]) <= gamma:
			x[i] = 0.
		if x[i] > gamma:
			x[i] = x[i] - gamma
		if x[i] < -gamma:
			x[i] = x[i] + gamma
	return x


# lr = learning rate
def lasso_grad(X, y, reg=1e-6, lr=1e-12, tol=1e-6, max_iters=3000, batch_size=300):
	# y = y.reshape(-1,1) # original y (making column vector)
	w = np.linalg.solve(X.T.dot(X), X.T.dot(y)) # solving for weights 'w'. This is the Normal Equation
	ind = np.random.randint(0, X.shape[0], size=batch_size)
	obj = [objective(X[ind], y[ind], w, reg=reg)]
	grad = grad_objective(X[ind], y[ind], w)
	while len(obj)-1 <= max_iters and np.linalg.norm(grad) > tol:
		ind = np.random.randint(0, X.shape[0], size=batch_size)
		grad = grad_objective(X[ind], y[ind], w)
		w = prox(w - lr * grad, reg*lr)
		obj.append(objective(X[ind], y[ind], w, reg=reg))

	return w, obj

def lasso_path(X, y, reg_min=1e-9, reg_max=1e-2, regs=10, **grad_args):
	W = np.zeros((X.shape[1], regs))
	tau = np.linspace(reg_min, reg_max, regs)
	for i in range(regs):
		W[:,i] = lasso_grad(X, y, reg=1/tau[i], max_iters=1000, batch_size=1024, **grad_args)[0].flatten()
	return tau, W

states = ["PA", "NC", "NV", "WI", "AZ", "IL"]
first = True
Categories = {}
CategoryL = []

# Going through CSV file in which I have grouped categories into general categories and putting this info in data structures
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


zip_info = {}
max_population = 0
max_poverty = 0.0
max_num_reviews = 0
max_income = 0

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
			zip_info[code] = {}
			zip_info[code]["population"] = population
			zip_info[code]["poverty"] = poverty
			zip_info[code]["income"] = income
			zip_info[code]["num reviews"] = []
			zip_info[code]["ratings"] = []
			zip_info[code]["avg rating"] = 0.0
			zip_info[code]["avg num reviews"] = 0.0
			if population > max_population:
				max_population = population
			if poverty > max_poverty:
				max_poverty = poverty
			if income > max_income:
				max_income = income


# categories -> zip codes -> business id ->
businesses = {}
businessX = {}

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

	if business_state != None and business_state in states and business_data["categories"] != [] and str(business_data["categories"][0]) in Categories and business_zip in zip_info:
		business_type = str(business_data["categories"][0]) # taking first category - might change later!
		global_category = Categories[business_type]

		#if global_category == "Restaurants & Bars":

		businessX[business_id] = {}
		businessX[business_id]["population"] = float(zip_info[business_zip]["population"]) / float(max_population)
		businessX[business_id]["poverty"] = float(zip_info[business_zip]["poverty"]) / float(max_poverty)
		businessX[business_id]["num_reviews"] = 0
		businessX[business_id]["income"] = float(zip_info[business_zip]["income"]) / float(max_income)
		businessX[business_id]["reviews"] = []
		businessX[business_id]["zip_code"] = business_zip
		for bus_type in CategoryL:
			if bus_type == global_category:
				businessX[business_id][bus_type] = 1
			else:
				businessX[business_id][bus_type] = 0

review_file = open('yelp_academic_dataset_review.json', 'r')
for line in review_file:
	review_data = json.loads(line)
	business_id = str(review_data["business_id"])
	rating = review_data["stars"]
	if business_id in businessX:
		businessX[business_id]["reviews"] += [float(rating) / 5.0]
		businessX[business_id]["num_reviews"] += 1
		zip_info[businessX[business_id]["zip_code"]]["ratings"] += [float(rating) / 5.0]
		if businessX[business_id]["num_reviews"] > max_num_reviews:
			max_num_reviews = businessX[business_id]["num_reviews"]

# normalizing review num
for store in businessX:
	businessX[store]["num_reviews"] = float(businessX[store]["num_reviews"]) / float(max_num_reviews)
	bus_zip = businessX[store]["zip_code"]
	zip_info[bus_zip]["num reviews"] += [businessX[store]["num_reviews"]]

for zc in zip_info:
	if float(len(zip_info[zc]["num reviews"])) != 0.0 and float(len(zip_info[zc]["ratings"])) != 0.0:
		zip_info[zc]["avg rating"] = float(sum(zip_info[zc]["ratings"])) / float(len(zip_info[zc]["ratings"]))
		zip_info[zc]["avg num reviews"] = float(sum(zip_info[zc]["num reviews"])) / float(len(zip_info[zc]["num reviews"]))


# Y = [average rating]
# X = [1, income, poverty, population, num_reviews, Restaurants & Bars, Retail, Home Services, Personal Services, 
#	Medical Services, Auto/Moto Services, Business Services, Party A/V Services, Financial Services, 
#	Education, Pet Services, Entertainment, Community, Sports & Recreation, Travel & Transportation]


X = []
Y = []

counter = 239

validation = []
validation_Y = []
test_count = 0
valid_count = 0

print zip_info

for zcode in zip_info:
	#print float(zip_info[zcode]["avg num reviews"])
	if float(zip_info[zcode]["avg num reviews"]) != 0.0:
		if counter != 0:
			test_count += 1
			counter -= 1
			entry = [1.0]
			entry += [float(zip_info[zcode]["income"]) / float(max_income)]
			entry += [float(zip_info[zcode]["poverty"]) / float(max_poverty)]
			entry += [float(zip_info[zcode]["population"]) / float(max_population)]
			entry += [float(zip_info[zcode]["avg num reviews"]) / float(max_num_reviews)]
			X += [entry]
			Y += [zip_info[zcode]["avg rating"]]
		else:
			valid_count += 1
			entry = [1.0]
			entry += [float(zip_info[zcode]["income"]) / float(max_income)]
			entry += [float(zip_info[zcode]["poverty"]) / float(max_poverty)]
			entry += [float(zip_info[zcode]["population"]) / float(max_population)]
			entry += [float(zip_info[zcode]["avg num reviews"]) / float(max_num_reviews)]
			validation += [entry]
			validation_Y += [zip_info[zcode]["avg rating"]]

"""

for bus in businessX:
	print businessX[bus]["Restaurants & Bars"]
	if float(businessX[bus]["num_reviews"]) != 0.0 and businessX[bus]["Personal Services"] == 1:
		if counter != 0:
			test_count += 1
			counter -= 1
			entry = [1.0]
			entry += [businessX[bus]["income"]]
			entry += [businessX[bus]["poverty"]]
			entry += [businessX[bus]["population"]]
			entry += [businessX[bus]["num_reviews"]]
			
			entry += [businessX[bus]["Restaurants & Bars"]]
			entry += [businessX[bus]["Retail"]]
			entry += [businessX[bus]["Home Services"]]
			#entry += [businessX[bus]["Personal Services"]]
			entry += [businessX[bus]["Medical Services"]]
			entry += [businessX[bus]["Auto/Moto Services"]]
			#entry += [businessX[bus]["Business Services"]]
			#entry += [businessX[bus]["Party A/V Services"]]
			entry += [businessX[bus]["Financial Services"]]
			#entry += [businessX[bus]["Education"]]
			entry += [businessX[bus]["Entertainment"]]
			#entry += [businessX[bus]["Community"]]
			#entry += [businessX[bus]["Sports & Recreation"]]
			entry += [businessX[bus]["Travel & Transportation"]]
			
			X += [entry]
			Y += [float(sum(businessX[bus]["reviews"])) / float(len(businessX[bus]["reviews"]))]
		else:
			valid_count += 1
			v_entry = [1.0]
			v_entry += [businessX[bus]["income"]]
			v_entry += [businessX[bus]["poverty"]]
			v_entry += [businessX[bus]["population"]]
			v_entry += [businessX[bus]["num_reviews"]]
			
			v_entry += [businessX[bus]["Restaurants & Bars"]]
			v_entry += [businessX[bus]["Retail"]]
			v_entry += [businessX[bus]["Home Services"]]
			#v_entry += [businessX[bus]["Personal Services"]]
			v_entry += [businessX[bus]["Medical Services"]]
			v_entry += [businessX[bus]["Auto/Moto Services"]]
			#v_entry += [businessX[bus]["Business Services"]]
			#v_entry += [businessX[bus]["Party A/V Services"]]
			v_entry += [businessX[bus]["Financial Services"]]
			#v_entry += [businessX[bus]["Education"]]
			v_entry += [businessX[bus]["Entertainment"]]
			#v_entry += [businessX[bus]["Community"]]
			#v_entry += [businessX[bus]["Sports & Recreation"]]
			v_entry += [businessX[bus]["Travel & Transportation"]]
			
			validation += [v_entry]
			validation_Y += [float(sum(businessX[bus]["reviews"])) / float(len(businessX[bus]["reviews"]))]

"""

print "test count: ", test_count
print "valid count: ", valid_count

real_Y = validation_Y

"""
# Correlation between each variable and avg rating for businesses
for i in range(1, len(X[0])):
	varL = []
	for j in range(len(X)):
		varL += [X[j][i]]
	#print "col " + str(i), correlation(varL, Y)
"""

# we instantiate our weights with the least squares estimate

XReal = np.matrix(X)

YReal = np.matrix(Y).T

XT = XReal.T

XTX = XT.dot(XReal)

XTY = XT.dot(YReal)

invXTX = XTX.I

Theta = invXTX.dot(XTY)

print "Theta: ", Theta

ThetaL = Theta.tolist()

#w, obj = lasso_grad(XReal, YReal)

#print w

#print "Original Theta: ", ThetaL
#print "Original objective: ", objective(XReal, YReal, Theta, 0.0)
first_w, first_obj = lasso_grad(XReal, YReal, reg=0.0)
#print "FIRST OBJ: ", first_obj[-1]
objectiveL = [first_obj[-1]]
regL = [0.0]
min_val = first_obj[-1]
min_reg = 0.0
regi = 0.0
min_w = first_w
for i in range(2,10):
	for j in range(9):
		regi = (1.0*float(j) + 1.0)*(10**(-float(i) - 1.0))
		regL += [regi]
		w, val = lasso_grad(XReal, YReal, reg=regi)
		print regi, val[-1]
		if val[-1] < min_val:
			min_val = val[-1]
			min_reg = regi
			min_w = w
		objectiveL += [val[-1]]

print "Min W: ", min_w

validation = np.matrix(validation)
validation_Y = np.matrix(validation_Y).T

validation_calc_y_theta = validation.dot(Theta)
#print "validation_calc_y_theta", validation_calc_y_theta.tolist()

validation_calc_y_minw = validation.dot(min_w)
#print "validation_calc_y_minw", validation_calc_y_minw.tolist()

validation_calc_y_theta_diff = abs(validation_calc_y_theta - validation_Y).tolist()
validation_calc_y_minw_diff = abs(validation_calc_y_minw - validation_Y).tolist()
yL = validation_Y.tolist()

acc_theta = []
acc_wmin= []
plot_y_theta = []
plot_y_wmin = []
y_theta_L = validation_calc_y_theta.tolist()
y_minw_L = validation_calc_y_minw.tolist()
theta_sum = 0.0
wmin_sum = 0.0
for q in range(len(validation_calc_y_theta_diff)):
	acc_theta += [float(validation_calc_y_theta_diff[q][0]) / float(yL[q][0])]
	acc_wmin += [float(validation_calc_y_minw_diff[q][0]) / float(yL[q][0])]
	theta_sum += float(validation_calc_y_theta_diff[q][0])
	wmin_sum += float(validation_calc_y_minw_diff[q][0])
	plot_y_theta += [y_theta_L[q][0]]
	plot_y_wmin += [y_minw_L[q][0]]

print "theta acc: ", sum(acc_theta) / float(len(acc_theta))
print "wmin acc: ", sum(acc_wmin) / float(len(acc_wmin))
print "theta avg rating diff: ", theta_sum / float(len(validation_calc_y_theta_diff))
print "wmin avg rating diff: ", wmin_sum / float(len(validation_calc_y_minw_diff))

plt.plot(real_Y, 'ro', plot_y_wmin, 'p-', plot_y_theta, 'g-')
plt.show()

"""
w1, obj1 = lasso_grad(XReal, YReal, reg = 1e3)

plt.plot(obj1)
plt.show()
"""

tau, W = lasso_path(XReal, YReal)
#W = W[1:]
#print "tau: ", tau
#print "W: ", W
#print "len: ", len(W)
final_weight = []
legendL = []
for i in range(len(W)):
	legendL += ["w" + str(i)]
	final_weight += [W[i][-1]]

print "NEW WEIGHTS:"
print final_weight

plt.plot(tau, W.T)
plt.legend(legendL)
plt.title("Lasso Path")
plt.show()


RXX = []
C = []

for i in range(1, len(X[0])):
	entry = []
	for j in range(1, len(X[0])):
		corrX1 = []
		corrX2 = []
		for row in X:
			corrX1 += [row[i]]
			corrX2 += [row[j]]
		entry += [correlation(corrX1, corrX2)]
	RXX += [entry]

for z in range(1, len(X[0])):
	corr = []
	for row in X:
		corr += [row[z]]
	C += [correlation(corr, Y)]


RXX = np.matrix(RXX)
C = np.matrix(C).T
CT = C.T

invRXX = RXX.I

R2 = CT.dot(invRXX.dot(C))
print R2