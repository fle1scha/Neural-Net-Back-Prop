### STA4026S - Analytics - CA2
### FLSANT005 - Antony Fleischer

dat_train = read.table('Rondebosch21km_2021_Train.txt', h = TRUE)
dat_val = read.table('Rondebosch21km_2021_Validate.txt', h = TRUE)
dat_test = read.table('Rondebosch21km_2021_Test.txt', h = TRUE)
attach(dat_train)

### Question 1
### =====================
## a)
#Five number summary
boxplot(Speed_21km, data = dat_train, ylab = "Average speed (scaled)", main = "Boxplot of Average Speed for Rondebosch Half Marathon", col = "cadetblue")
summary(Speed_21km)
subset(dat_train, dat_train$Speed_21km > 17)
abline(h = mean(Speed_21km), col = "red", lwd = 2)

#Distribution and skewness
par(mfrow = c(1, 1))
hist(Speed_21km, xlab = "Average speed (scaled)", main = "Distribution of Average Speed for Rondebosch Half Marathon")
rug(Speed_21km)

#Speed vs Sex
par(mfrow = c(1, 2))
plot(Speed_21km ~ Sex, col = "cadetblue", main= "Average Speed vs Sex", ylab = "Average speed (scaled)")
cor(Speed_21km, as.numeric(Sex))

mean(Speed_21km)
mean(subset(Speed_21km, Sex == 'Male'))
mean(subset(Speed_21km, Sex == 'Female'))

#Speed vs Shoe
plot(Speed_21km ~ ShoeBrand, col  = "cadetblue", main = "Average Speed vs Shoe", ylab = "Average speed (scaled)")
cor(Speed_21km, as.numeric(ShoeBrand))

#Speed vs Age
plot(Speed_21km ~ Age_Scl, main = "Average Speed vs Age", ylab = "Average speed (scaled)", xlab = "Age (scaled)")
age_lm = lm(Speed_21km~Age_Scl)
abline(age_lm, col = 'red', lwd = 2)
cor(Speed_21km, Age_Scl)


#Speed vs Nutrition
plot(Speed_21km ~ Nutrition,  main = "Average Speed vs Nutrition", ylab = "Average speed (scaled)", xlab = "Nutrtion")
age_lm = lm(Speed_21km~Nutrition)
abline(age_lm, col = 'red', lwd = 2)
cor(Speed_21km, Nutrition)

## b)
X_train = matrix(cbind(Nutrition,Age_Scl,as.numeric(Sex)-1, as.numeric(ShoeBrand)-1), ncol = 4)
Y_train = matrix(dat_train$Speed_21km, ncol = 1)

# Specify activation functions for the hidden and output layers:
sig1 = function(x) #Logistic activation function for hidden layers
{
  1/(1+exp(-x))
}

sig2 = function(x) #Logistic activation function for output layer.
{
  x
}

g = function(Yhat,Y) #Cross-entropy cost function
{
  0.5*(Yhat-Y)^2
}

neural_net = function(X,Y,theta, m, lam)
{
  # Relevant dimensional variables:
  N = dim(X)[1]
  p = dim(X)[2]
  q = dim(Y)[2]
  
  # Indexing of theta parameter vector
  index = 1:(p*m)
  W1    = matrix(theta[index],p,m)
  index = max(index) + 1:(m*q)
  W2    = matrix(theta[index],m,q)
  index = max(index) + 1:m
  b1    = matrix(theta[index],m,1)
  index = max(index) + 1:q
  b2    = matrix(theta[index],q,1)
  out   = rep(0,N)
  error = rep(0,N)
  
  #Matrix Approach
  S1 = X%*%W1
  b1_t = rep(b1, dim(X)[1])
  b1_t = matrix(t(b1_t), nrow = dim(X)[1], byrow = TRUE)
  S1 = S1 + b1_t
  Z1 = sig1(S1)
  
  S2 = Z1%*%W2
  b2_t = rep(b2, dim(X)[1])
  b2_t = matrix(t(b2_t), nrow = dim(X)[1], byrow = TRUE)
  S2 = S2 + b2_t
  Z2 = sig2(S2)
  
  E1 = (0.5*sum((Z2-Y)^2))/N
  E2 = E1 + lam/N*(sum(W1^2)+sum(W2^2))
  
  return(list(out = Z2, E1 = E1, E2 = E2))
}

obj = function(pars)
{
  res = neural_net(X, Y, pars, m, l)
  return(res$E2)
}

## c)
#Most work here is done in the neural_net function
par(mfrow = c(1,1))

p = dim(X_train)[2]
q = dim(Y_train)[2]
m = 4
np = p*m+m*q+m+q
theta = runif(np,-2,2)
nn = neural_net(X_train, Y_train, theta, m, 0)

## d)
X_val = matrix(cbind(dat_val$Nutrition, dat_val$Age_Scl, as.numeric(dat_val$Sex)-1, as.numeric(dat_val$ShoeBrand)-1), ncol = 4)
Y_val  = matrix(dat_val$Speed_21km, ncol = 1)

#Number of params for 3-network.
a = 3
np_a = p*a+a*q+a+q

#Number of params for 5-network.
b = 5
np_b = p*b+b*q+b+q

#Create empty variables to store min errors, respective lambda values and weight estimates.
#Define lambda sequence
lams    = c(0,exp(seq(-10,-2, length =15)))

ValError_A = rep(NA,length(lams)) 
ValError_B   = rep(NA,length(lams))
estimateA = NA
estimateB = NA
minErrorA = 999
minErrorB = 999
lambdaA = NA
lambdaB = NA


 for(i in 1:16)
 {
   X = X_train
   Y = Y_train
   l = lams[i]
   theta_a = runif(np_a,-1,1)
   theta_b = runif(np_b, -1,1)
   m = a
   res_opt_a   = nlm(obj,theta_a, iterlim = 400)
   m = b
   res_opt_b   = nlm(obj,theta_b, iterlim = 400)
   X = X_val
   Y = Y_val
   resA = neural_net(X_val,Y_val,res_opt_a$estimate,3,lams[i])
   resB = neural_net(X_val,Y_val,res_opt_b$estimate,5,lams[i])
   ValError_A[i] = resA$E2
   ValError_B[i] = resB$E2


   # plot(Y_val)
   # points(resA$out, col = 'dodgerblue1')
   # points(resB$out, col = 'orange')
   #
   # print(res_opt_a$code)
  if (resA$E2 < minErrorA)
   {
    print("Min error found A")
    print(i)
     print(resA$E2)
     minErrorA = resA$E2
     estimateA = res_opt_a$estimate #Usi
     lambdaA = lams[i]
   }
   if (resB$E2 < minErrorB)
   {
     print("Min error found B")
      print(i)
    # print(res_opt_b$estimate)
     print(resB$E2)
     minErrorB = resB$E2
     estimateB = res_opt_b$estimate
     lambdaB = lams[i]
   }
  
   
   
}

resB = neural_net(X_val,Y_val,res_opt_a$estimate,3,lams[16])
resC = neural_net(X_val,Y_val,estimateA,3,lambdaA)
resB$E2
resC$E2

#Min and max values help when plotting val_error vs lambda.
min = min(c(minErrorA, minErrorB))
max = max(c(max(ValError_A), max(ValError_B)))
plot(ValError_A~lams, type = 'l', col = 'dodgerblue1', 
     ylim = c(min, max), 
     lwd = 1,
     ylab = "Validation Error",
     xlab = "Lambda",
     main = "Validation Error of Neural Networks")

lines(ValError_B~lams, lwd = 1, col = 'orange')
legend(x = 'topright', legend=c("3-NN", "5-NN"), 
       fill = c("dodgerblue1","orange"))


which.min(c(minErrorA, minErrorB))
minErrorA
lambdaA

## e)
#Define male Nike lattice.
male = subset(dat_train, (dat_train$ShoeBrand=='Nike'))
male = subset(male, male$Sex=='Male')

M = 200
x1 = seq(min(male$Nutrition), max(male$Nutrition), length = M)
x2 = seq(min(male$Age_Scl), max(male$Age_Scl), length = M)
N = rep(x1, M)
A = rep(x2, each = M)
lat = data.frame(N = N, A = A, S = 1, Sh = 1)
Ylat =  cbind(rep(x2*0, each = M))

#Fit the lattice to the model and return predictions. 
res_fitted = neural_net(as.matrix(lat), Ylat, estimateA, 3, lambdaA)
z = matrix(res_fitted$out, nrow = M, ncol = M)
z
filled.contour(x = seq(min(male$Nutrition), max(male$Nutrition), length = M),
               y = seq(min(male$Age_Scl), max(male$Age_Scl), length = M),
               z,
               color = function(n) hcl.colors(n, "Blue-Yellow"),
               nlevels = 30,
               #zlim = range(min(male$Speed_21km), max(male$Speed_21km)),
               plot.title = title(main = "Avg Speed of Male Nike Athletes",
                                  xlab = "Nutrition", ylab = "Age (scaled)"))



#Define female Nike lattice.
female = subset(dat_train, (dat_train$ShoeBrand=='Nike'))
female = subset(female, female$Sex=='Female')

M = 200
x1 = seq(min(female$Nutrition), max(female$Nutrition), length = M)
x2 = seq(min(female$Age_Scl), max(female$Age_Scl), length = M)
N = rep(x1, M)
A = rep(x2, each = M)
lat = data.frame(N = N, A = A, S = 0, Sh = 1)
Ylat =  cbind(rep(x2*0, each = M))

res_fitted_female = neural_net(as.matrix(lat), Ylat, estimateA, 3, lambdaA)
z = matrix(res_fitted_female$out, nrow = M, ncol = M)

filled.contour(x = seq(min(female$Nutrition), max(female$Nutrition), length = M),
               y = seq(min(female$Age_Scl), max(female$Age_Scl), length = M),
               z,
               color = function(n) hcl.colors(n, "YlOrBr"),
               nlevels = 30,
               zlim = range(min(female$Speed_21km), max(female$Speed_21km)),
               plot.title = title(main = "Avg Speed of Female Nike Athletes",
                                  xlab = "Nutrition", ylab = "Age (scaled)"))


## f)
#Predictions saved to .csv file.
X_test = matrix(cbind(dat_test$Nutrition, dat_test$Age_Scl, as.numeric(dat_test$Sex)-1, as.numeric(dat_test$ShoeBrand)-1), ncol = 4)
Y_test = matrix(rep(0, dim(X_test)[1]), ncol = 1)
X_test
Y_test
test_model = neural_net(X_test, Y_test, estimateA, 3, lambdaA)

pred = data.frame(predictions = matrix(test_model$out, ncol = 1))
write.table(pred,'FLSANT005_STA4026S_CA2.csv', quote = F, row.names = F, sep = ',')


### Question 2


# Let's fake a dataset and see if the network evaluates:
set.seed(2020)
N = 50
x = runif(N,-1,1)
e = rnorm(N,0,1)
y = 2*sin(3*pi*x)+e
plot(y~x, pch = 16, col = 'blue')

# Get the data in matrix form:
X = matrix(x,N,1)
Y = matrix(y,N,1)

# Specify activation functions for the hidden and output layers:
sig1 = function(x)
{
  1/(1+exp(-x))
}
sig1. = function(x)
{
  1/(1+exp(-x))*(1-1/(1+exp(-x)))
}
sig2 = function(x)
{
  x
}
sig2. = function(x)
{
  1+0*x
}

g = function(Yhat,Y)
{
  0.5*(Yhat-Y)^2
}
g. = function(Yhat,Y)
{
  (Yhat-Y)
}
# Write a function that evaluates the neural network (forward recursion):
# X     - Input matrix (N x p)
# Y     - Output matrix(N x q)
# theta - A parameter vector (all of the parameters)
# m     - Number of nodes on hidden layer
# lam   - Regularisation parameter (see later)
neural_net = function(X,Y,theta, m, lam)
{
  # Relevant dimensional variables:
  N = dim(X)[1]
  p = dim(X)[2]
  q = dim(Y)[2]
  
  # Populate weight-matrix and bias vectors:
  index = 1:(p*m)
  index
  W1    = matrix(theta[index],p,m)
  W1
  index = max(index) + 1:(m*q)
  W2    = matrix(theta[index],m,q)
  W2
  index = max(index) + 1:m
  b1    = matrix(theta[index],m,1)
  index = max(index) + 1:q
  b2    = matrix(theta[index],q,1)
  
  # Evaluate network:
  out   = rep(0,N)
  error = rep(0,N)
  
  dW1 = W1*0
  dW2 = W2*0
  dX1 = X*0
  db1 = b1*0
  db2 = b2*0
  
  i = 1
  for(i in 1:N)
  {
    a0 = matrix(X[i,],ncol = 1)
    z1 = t(W1)%*%a0+b1
    a1 = sig1(z1)
    z2 = t(W2)%*%a1+b2
    a2 = sig2(z2)
    
    d2 = g.(a2,Y[i])*sig2.(z2) #working gradient of layer 2 derivative of C * sigma(z2)
    d1 = (W2%*%d2)*sig1.(z1) #working gradient of layer 1
    d0 = (W1%*%d1) #working gradient of layer 0 (inputs)
    db2 = db2+d2
    db1 = db1+d1
    dX1[i] = d0 #assign
    dW2 = dW2+a1%*%t(d2)
    dW1 = dW1+a0%*%t(d1)
    
    out[i]   = a2
    error[i] = g(a2,Y[i])
  }
  
  # Calculate error:
  E1 = sum(error)/N
  #E2 = ...
  
  # Return predictions and error:
  return(list(out = out, E1 = E1, grad = c(dW1,dW2,db1,db2)/N, gradX = dX1))
}



# We need to know the number of parameters in the network:
m          = 10
p          = dim(X)[2]
q          = dim(Y)[2]
npars      = p*m+m*q+m+q
npars
theta_rand = runif(npars,-2,2)
theta = theta_rand
res        = neural_net(X,Y,theta_rand,m,0)

# Set an objective and minimize
obj = function(pars)
{
  res = neural_net(X,Y,pars,m,0)
  return(res$E1)
}


res_opt = nlm(obj,theta_rand, iterlim = 250)


# res_rc     = neural_net(x_lat,y_dummy,res_opt$estimate,m,0)
# 
# # Plot response curve and fitted values
# plot(y~x,pch = 16,col = 'blue')
# points(res_fitted$out~X,col = 'red')
# lines(res_rc$out~x_lat)


#====================================================
# Continue here:
#====================================================

## a)
x_lat = cbind(seq(-1,1,1/100))
y_dummy = cbind(x_lat*0)
h     = 0.01
grad_check = c()
for(k in 1:length(x_lat))
{
  x_kp = x_lat
  x_km = x_lat
  x_kp[k] = x_kp[k]+h/2
  x_km[k] = x_km[k]-h/2
  
  res_kp  =  neural_net(x_kp,y_dummy,res_opt$estimate,m,0)
  res_km  =  neural_net(x_km,y_dummy,res_opt$estimate,m,0)
  
  grad_check[k] = (res_kp$out[k]-res_km$out[k])/h
}

#The partial derivative of the cost function with respect to X. So simple!
CpartialX = 6*pi*cos(3*pi*x_lat) 

#Plot approximate gradient vs regularly spaced x
res_fitted = neural_net(x_lat, y_dummy, res_opt$estimate, 10, 0)
plot(res_fitted$gradX~x_lat, type = 'l', lwd = 2, col = 'dodgerblue1', xlab = 'X', ylab = 'Gradient', main = "Gradient of C with respects to X")
#add actual cost function derivative/gradient
lines(x_lat, CpartialX, lwd = 2, col = 'orange')

lines(grad_check~x_lat, type = 'l', lwd = 2, col = 'darkgreen')
legend(x = 'bottomright', legend=c("Backprop", "True Deriv", "Approximation"), 
       fill = c("dodgerblue1","orange", "darkgreen"))

               