# machine-learning-exercise-3-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning Exercise 3 Solved](https://www.ankitcodinghub.com/product/machine-learning-labs/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;110191&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning Exercise 3 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
www.epfl.ch/labs/mlo/machine-learning-cs-433

Goals. The goal of this exercise is to

Implement and debug least-squares.

Implement, debug and visualize basis function models.

Understand overfitting.

Implement ridge regression.

Setup, data and sample code. Obtain the folder labs/ex03 of the course github repository

github.com/epfml/ML course

We will continue to use the dataset height weight genders.csv as well as a new dataset dataEx3.csv in this exercise. We have provided sample code that already contains useful snippets of code required for this exercise.

You will be working in the notebook ex03.ipynb for the exercises of this week, by filling in the corresponding functions in the provided template code.

1 Least Squares and Linear Basis Functions Models

1.1 Least squares

Exercise 1:

Fill in the notebook function least squares(y, tx) which implements the solution of the normal equations as discussed in the class. This function should return the optimal weights, and the mean-squared error.

Hint: You should not try to solve a linear system Ax = b by using the numpy.linalg.inv function.

To debug your code, you can use the output of the last exercise. Run gradient descent or grid search on the height-weight data from the last exercise, and make sure you get a similar resulting w vector using all three methods.

This is a useful method to debug your code, i.e. first implementing a simple method and then using it to check more complicated methods. If you have not finished Exercise 2, please first finish implementing the grid search method. If you are lagging behind, do not worry. You will get the opportunity to catch up later, but it is important that you eventually take time to finish previous exercises.

1.2 Least squares with a linear basis function model

We will now implement and visualize a basis function model for the data dataEx3.csv.

As explained in the class, linear regression might not be directly suitable for nonlinear data. We will use polynomial basis functions to fit nonlinear data.

ϕj(x) := xj (1)

As we have seen in the lecture notes, the technique of feature expansion by the linear basis function model does allow us to still use linear regression techniques, to fit nonlinear data (recall that in our first simple setting, we assume that each input point is just one real value). As a result, we will be able to fit the data using different degrees of polynomials, e.g. a degree two polynomial (which is a linear combination of 1, x and x2), or a degree three polynomial (which is a linear combination of 1, x, x2 and x3), etc.. Higher degree polynomials are more expensive to compute and to fit, but can capture finer details in the data, which results in more expressive models. Think about the pros and cons of choosing a very high or very low degree.

To measure the fit of our model, we will use a cost function called the Root-Mean-Square-Error (RMSE). It is related to MSE as follows:

RMSE(w) := p2 · MSE(w) (2)

The magnitude of MSE can be difficult to interpret since it involves a square, while RMSE provides a more interpretable measure on the same scale as the error of one point. There are better measures in terms of statistical properties, like R2, but we don’t need these for now. See the book “Introduction to Statistical learning” if you’re interested in more details.

Let us now implement polynomial regression, using the technique of linear basis functions, and visualize the predictions.

Exercise 2:

The goal of this exercise is to plot the data along with predictions using polynomial regression. Your goal is to find a good w using polynomial regression, when using polynomials of degrees 1, 3, 7, and 12 respectively. You might want to reuse the function from the previous exercise to calculate the RMSE.

Fill in the notebook function build poly(x, degree). The input of this function is the vector of the data examples xn ∈ R for 1 ≤ n ≤ N. As an output, the function must return the extended feature matrix

ϕ(x1)

..

 

Φe := ϕ(xn) where

 ..  ϕ(xN)

that is the matrix formed by applying the polynomial basis functions to all input data, for the degree of j = 0 up to j =degree.

When finished, you must COPY your implementation to the separate file build polynomial.py for the plot function to work.

Fill in the notebook function polynomial regression(). If the code runs successfully, you will see the data and the fit. You will clearly see why linear regression is not a good fit, while polynomial regression produces a better fit.

You can see that RMSE decreases as we increase the degree of the polynomial. Does it mean that the fit gets better as we increase the degree? Which fit is the best in your view?

2 Evaluating Model Prediction Performance

The answer to the last question should be clear if you followed the lecture. If not, discuss with others and clarify.

In practice, it matters that predictions are good for unseen examples, not only for training examples. To simulate the reality, we will now split our dataset into two parts: training and testing. We will fit the data using training data and compute RMSE on both test and training data.

Exercise 3:

The notebook function train test split demo() is supposed to show the train and test splits for various polynomial degrees.

To split the data, please fill in the notebook function split data(x, y, ratio, …). Do you think that the order of samples is important when doing the split?

Fill in the notebook function train test split demo(). If the code runs successfully, you will see RMSE values printed for degrees 3, 7 and 12. For each degree, there are again three RMSE values which correspond to the following three splits of the data.

– 90% training, 10% testing

– 50% training, 50% testing

– 10% training, 90% testing

Look at the training and test RMSE for degree 3. Does this makes sense? Why? Discuss with others if you are unclear.

Now look at RMSE for other two degrees. Do these make sense? Why? Discuss with others if you are unclear.

Which split is better? Why?

The test RMSE for degree 12 is ridiculously high for the split 10%-90%. Why do you think this is the case? The answer lies in numerical inaccuracies. Make sure you understand this.

BONUS: Imagine you have 5000 samples instead of 50. Which split might be better in that situation?

3 Ridge Regression

The previous exercise shows overfitting when using complex models. Let us now correct it using Ridge Regression, defined as

Fill in the notebook function ridge regression(). You can debug your code by setting λ = 0. This should essentially give the same answer as least-squares code. You can also check that for large value of lambda, RMSE should be really bad.

Play with the demo ridge regression demo() by choosing a split of 50%-50% and plot train and test errors vs λ for polynomial degree 7. You should get a similar plot as Figure 1.

Figure 1: Ridge Regression Demo.

Theory Exercises

1. Warm-Up

(a) Show that the sum of two convex functions is convex.

Hint: use the definition of convexity,

f : X → R is convex ⇔ ∀x,y ∈ X,∀λ ∈ [0,1] : f(λx + (1 − λ)y) ≤ λf(x) + (1 − λ)f(y).

(b) How do you solve the linear system Ax = b? When is it not possible, and why?

Hint: Invertible matrix

(c) What is the computational complexity of

Grid search?

(one step of) Gradient Descent for linear regression with MSE cost?

(one step of) Stochastic Gradient Descent for linear regression with MSE cost?

If needed, refresh your memory of the complexity of algebraic operations.

(d) Consider a problem with two input variables, x = (x1,x2), and one output variable y. Given the two samples below, find the coefficients w = (w1,w2) of the linear relationship x⊤w = w1x1+w2x2 = y.

x1 x2 y

Sample 1 400 -201 200

Sample 2 -800 401 -200

Do the exercise again, but with a slight change in the inputs: x1 for sample 1 is now 401 instead of 400

x1 x2 y

Sample 1 401 -201 200

Sample 2 -800 401 -200

Compare the resulting w = (w1,w2) for both cases. Familiarize yourself with the concept of condition number as a way to diagnose ill-conditionning. You can find condition number calculators online or use numpy.linalg.cond).

2. Cost functions

A cost function defines how you evaluate a solution, and you might have different requirements depending on the problem. Using the MSE, if a your model makes an error of 5 on a sample, you add 25/2 to the cost of your model, regardless of the target. You might want to penalize this differently if you care about the relative error; an output of 1005 when 1000 was expected might be OK, but mistaking a 6 for a 1 might not. In this case, you can use a function that takes the relative error of the target yn into account, like this one:

.

Where f is the model and ϵ is a small constant to avoid divisions by zero. Note that we have defined the cost function per example here. You can imagine the total cost function being defined as .

(a) Try the function on some [prediction, target] pairs, or plot it, to see how it behaves (by hand or using Python or the wolfram alpha website, no need to code)

(b) Compute its gradient, assuming a standard linear regression f(xn,w) := x⊤nw

(c) How would you implement the gradient? Again, no need to code – try to find a formula using standard matrix operations, along with element-wise multiplication and summation/product over columns/rows.

(d) How sensitive is this function to outliers? Compare two cases: the target is 1, but in one case our model assigns it 10, and in the other 100. (e.g. with ε = 1) How does the error changes? Compare with the following cost function, for the n data example:

.

Note: The higher the error on a sample is, relative to the other samples, the more your model will try to fit this sample.
