# Machine Learning Engineer Nanodegree
## Capstone Proposal
Cassandra Carr Sugerman 
January 15, 2018

## Proposal

### Domain Background

This machine learning capstone project seeks to predict domestic flight delays and highlight domestic flight trends. With the air transportation industry continuing to increase, flight delays continue to be a problem. For the consumer, a flight delay leads to lost travel time and overall dissatifaction in the airline. For the airline, a flight delay usually starts a chain of other delays leading to decreased efficiency and many unssatisfied customers. I believe everyone, including myself, has been put in a situation where a flight delay or cancellation caused major setbacks for a trip. 

Previous research to solve this problem included a project by Gopalakrishnan and Balakrishnan where they looked at several algorithms for flight delay prediction. The team used inputs such as time of day, day of weak, season, previous Origin-Destination pair delays and delays of adjacent Origin-Destination pairs. They were able to achieve the best classification results using an Artifical Neural Network and the best regression results using a Markov Jump Linear System (a specialized hybrid system model of network delay propagation). 

Research by Ding was also performed. He looked at developing a classification algorithm for predicting delays over 30 minutes. Ding gathered time, airport, airline, flight features, airplane features, and weather data. However, his algorithm ended up being relatively simple and only used departure delay time and flight distance to output arrival delay time. He compared multiple linear regression and Naive Bayes to determine the better classifier, multiple linear regression had better results. 

In addition, a project by Cole was performed to predict flight delays. Cole set out to develop a classifier to predict delays greater than 15 minutes. He used time of year, day, month, flight duration, origin airport, and airline as inputs. The team used logistic regression to develop their algorithm and achieved an Area Under the Receiver Operating Characteristic curve (AUC) of 0.689. AUC was used for evaluating the model since there was a skewed dataset of delayed vs non-delayed flights. Their argument was that these results were very good since weather and other flight delays were not taken into consideration. These circumstances would be most useful with booking a flight since weather and flight delays are not known at this time as well. 

Resources:

Gopalakrishnan, Karthik & Balakrishnan, Hamsa (2017). A Comparative Analysis of Models for Predicting Delays in Air Traffic Networks. [http://web.mit.edu/hamsa/www/pubs/GopalakrishnanBalakrishnanATM2017.pdf](http://web.mit.edu/hamsa/www/pubs/GopalakrishnanBalakrishnanATM2017.pdf)

Ding, Yi (2017). Predicting flight delay based on multiple linear regression. [http://iopscience.iop.org/article/10.1088/1755-1315/81/1/012198/pdf](http://iopscience.iop.org/article/10.1088/1755-1315/81/1/012198/pdf)

Cole, Scott (2017). Delays of US domestic flights: trends and predictability. [https://srcole.github.io/2017/04/02/flight_delay/](https://srcole.github.io/2017/04/02/flight_delay/)

### Problem Statement

This project's goal is to predict whether or not a flight will be delayed longer than 15 minutes. Flight delays longer than 15 minutes usually result in a ripple affect of missed flights and further delayed flights. This information could be used by customers to better choose and plan their flights. It could also be used by airports and airlines to identify areas of improvement and better plan for delays. 


### Datasets and Inputs

I plan to look at addressing the problem for (2) scenarios:
1. Long term prediction - In this instance only base flight information will be used to predict delays, weather or other flight interactions will not be considered. This would be for the scenario when a customer is booking their flight and they would like to choose a flight with a low probablity of being delayed.
2. Short term prediction - In this instance base flight information will be used, but in additon, weather data and other flight interactions will be considered. This would be for the scenario when a customer would like to see the likelyhood their flight will be delayed 2 hours before the scheduled flight time. 

The following data will be used as inputs:
1. Month
2. Day of the week
3. Day of the month
4. Airline 
5. Origin Airport
6. Destination Airport
7. Scheduled departure time
8. Flight time
9. Average arrival delay of all origin airport flights 4-2 hours before the flight in question
10. Average departure delay of all destination airport flights 4-2 hours before the flight in question
11. Average origin airport temperature 6-2 hours before the flight in question
12. Average destination airport temperature 6-2 hours before the flight in question
13. Total precipitation at origin airport 8-2 hours before the flight in question
14. Total precipitation at destination airport 8-2 hours before the flight in question
15. Average wind gust at origin airport 3-2 hours before the flight in question
16. Average wind gust at destination airport 3-2 hours before the flight in question

Data inputs 1 - 10 will be supplied from the [Bureau of Transportation Statistics](https://www.transtats.bts.gov/DL_SelectFields.asp). 

Data inputs 11-16 will be supplied from the [Iowa State University ASOS dataset](https://mesonet.agron.iastate.edu/request/download.phtml).

Data inputs 1-8 will be initially tested for scenario #1 (long term prediction). Then, data inputs 9-16 will be added for scenario #2 (short term prediction).

The data concerning flight information (month, day of the week, day of the month, departure time) will be used to see if there are increased flight delay trends with certain months or days. The flight time is used to see if there are trends between the length of the flight and flight delays. The airline data is used to see if one airline delays more flights than others. The origin and destination airports are used to see if certain airports have more delays than others. The weather data is used to see the link between weather and flight delays. The average delay times for origin airport flights and destination airport flights are used because flight delays tend to cause a ripple effect and this data will allow us to look at how much this influences the probability of a flight being delayed. 

The model result, used for training and testing, will be defined as a flight delay greater than 15 minutes. A cancellation will fall under a delay greater than 15 minutes. This informaiton will come from the [Bureau of Transportation Statistics](https://www.transtats.bts.gov/DL_SelectFields.asp). 

### Solution Statement

The solution for this project is a classification model that predicts if a flight will be delayed greater than 15 minutes, based on the inputs defined above. The classification model developed will be trained and tested using seperate, defined, datasets. It will be measured by how the test set performs, based on accuracy and Area Under the Receiver Operating Characteristic Curve (AUC).

### Benchmark Model

In the Domain Background section, defined above, (3) previous projects aimed at predicting flight delays were described. It is my goal to take the most successful aspects of the (3) projects in order to more effectively solve the problem. All (3) projects developed a classification model with some defined amount of time constituting delayed or not delayed. I have chosen to use 15 minutes since delays longer than this typically have repercussions. From research by Gopalakrishnan and Balakrishnan, I decided to take into consideration related flight delays. From Ding, I determined it would be useful to look at weather data. And from Cole, I came to the conclusion that the best way to look at this problem is from (2) scenarios: a long term perpective and short term perspective, as I have described above.

Based on the models used in each of the (3) projects described, I have narrowed down the models to test:
1. Artificial Neural Network (from Gopalakrishnan and Balakrishnan) - This model had the best mean error when looking the classification of flight delays during their research
2. Logistic regression, SVMs, and Random Forest (from Cole) - These models all had similar results during Cole's study, they did not spend enough time fine tuning parameters to see which would give the best results

During testing, Cole used  Area Under the Receiver Operating Characteristic curve (AUC) to evaluate their model. They achieved a result of of 0.689. 

### Evaluation Metrics

The evaluation metric used will be AUC, which was the same used in the benchmark model done by Cole. Since the data for flight delays is skewed, with the majority of flights not being delayed, accuracy is not a good indicator of performance. By looking at AUC, we put emphasis on true positive rate vs false positive rate. 

AUC is area under the receiver operating characteristic curve, which is a plot of the true positive rate against the false positive rate. 

True Positive Rate = (True Positives) / (True Positives + False Negatives)
False Positive Rate = (False Positives) / (False Positives + True Negatives)

### Project Design

##### Step 1

