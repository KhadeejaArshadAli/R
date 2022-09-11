#Name: Khadeeja Arshad Ali
#Class: BSCS Sec:A
#Project: Statistics
#Teacher: Dr.Tasheen Jillani
#Output: Word Document



# Data Set
#I went for the titanic data set from Kaggle
#It provides the information on the fate of passengers on the Titanic, summarized according to economic status (class), sex, age and survival.

#Importing the dataset by using the import function
setwd("C:/Users/RS Laprop/Downloads/titanic")

#Loading The Necessary Libraries
library(ggplot2)
library(caret)
library(randomForest)
library(e1071)
library(grid)
library(gridExtra)
library(mice)
library(dplyr)
library(pscl)

#Loading the training and testing dataset
train <- read.csv(file = "train.csv",na.strings = c("N/A","DIV/0!",""), stringsAsFactors = FALSE)
test <- read.csv(file="test.csv",na.strings = c("N/A","DIV/0!",""),stringsAsFactors = FALSE)
titanic<- bind_rows(train,test)

#Understanding the basic properties of the training dataset
summary(titanic)
dim(titanic)
str(titanic)
#It can be seen that there are some missing values and "NA" Values in the dataset. 
#We will tabulate the "NA" values.
titanic[titanic==""] <- NA
a <- apply(titanic,2,is.na)
summary(a)
apply(a,2,sum)
#It can be seen that Age,Cabin and Embarked variables have missing values. 
#Cabin has most number of missing values.
#These missing values can be found by using 'Multivariate Imputation by Chained Equations (MICE)' package.


#Feature Engineering
#"Name" Column: It can be seen that there is a column "Name" in the dataset having names of each passenger.
# These individual names may not useful for prediction 
#but titles in each names (eg.Mr., Mrs.,) 
#can given better insights into the passenger profile and can be used for model building.
titanic$Salutation <- gsub('(.*, )|(\\..*)', '',titanic$Name)
table(titanic$Sex,titanic$Salutation)
#It can be seen that only 'Mr.','Mrs.','Miss.','Master" are prominent titles. 
#We could group synonyoms titles and lables others as "Misc" as follows,

misc <- c("Capt","Col","Don","Dr","Jonkheer","Lady","Major","Rev","Sir","the Countess","Dona")
titanic$Salutation[titanic$Salutation == "Mlle"] <- "Miss"
titanic$Salutation[titanic$Salutation == "Mme"] <- "Miss"
titanic$Salutation[titanic$Salutation %in% misc] <- "Misc"
table(titanic$Sex,titanic$Salutation)

#Separating Surname from the given name in the dataset
titanic$Surname <- sapply(titanic$Name,function(x) strsplit(x, split = '[,.]')[[1]][1])
s <- nlevels(factor(titanic$Surname))
paste('We have', s, 'unique surnames in the titanic dataset amongst',nrow(titanic), 'passangers.')

#Determining the family size and creating family variable
titanic$FamilySize <- titanic$SibSp + titanic$Parch + 1
titanic$Family <- paste(titanic$Surname,'-',titanic$familysize)

#Creating a new variable to identify the deck of the passanger. 
#Deck of the passenger can be found by using the first character of his cabin number.
titanic$Deck <- substr(titanic$Cabin,1,1)
paste("Titanic has", nlevels(factor(titanic$Deck)),"decks on the ship.")

#With new feature engineered from existing columns we could then predict and fill the missing values. 
#Before missing value imputation the character variables are converted into factor variables where levels could be used for better prediction than characters.
titanicreengineered <- titanic[,c("Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Salutation","Deck","FamilySize","Family")]
list <- c("Survived","Pclass","Sex","Embarked","Salutation","Deck","Family")
titanicreengineered[list] <- lapply(titanicreengineered[list],function(x) as.factor(x))
str(titanicreengineered)


#Missing Value Imputation
#This is done with "random forest" as imputation method as it could work well with predication involving categorical variables. 
#New complete dataset is stored in the name of "titanicreengineered".
table(titanicreengineered$Sex,titanicreengineered$Survived)

set.seed(6)
imp= mice(titanicreengineered, method = "rf", m=5,rfPackage = "randomForest")
imputedtitanicreengineered = complete(imp)
summary(imp)

#Verifying the quality of imputation for any missing value which still remains unimputed.
apply(apply(imputedtitanicreengineered,2,is.na),2,sum)

#We can check the results before imputation & post imputation for any of the parameters, let's say 'Fare'
par(mfrow=c(1,2))
hist(titanicreengineered$Fare, main = "Before Imputation", col = "violet")
hist(imputedtitanicreengineered$Fare, main = "Post Imputation", col = "blue")
#mutate 0/1 to died and survied
#As distribution pattern remains identical before and post imputation, quality of imputation is excellent.

#Data Visualization

#Let us try to establish relationship between Family Size and Survived people.
ggplot(imputedtitanicreengineered,aes(x=FamilySize, fill = factor(Survived))) +
  geom_bar(stat = "count", position = "dodge") +
  scale_x_continuous(breaks = c(1:11)) +
  labs( x= "Familysize")
#Result:We can conclude that single passenger survival rate is highest.

# we can group the family size as 'Single', 'Small Size Family' & 'Large Size Family'
'Single' -> imputedtitanicreengineered$FamilySizeC[imputedtitanicreengineered$FamilySize == 1]
'Small' -> imputedtitanicreengineered$FamilySizeC[imputedtitanicreengineered$FamilySize > 1 & imputedtitanicreengineered$FamilySize < 5]
'Large' -> imputedtitanicreengineered$FamilySizeC[imputedtitanicreengineered$FamilySize >4]
mosaicplot(table(imputedtitanicreengineered$FamilySizeC,imputedtitanicreengineered$Survived), main = "Survival by Family Size", shade = TRUE)
#Result:It can easily be inferred that Small Families have the highest survival rate amongst the single and Large Families.

#Now, we'll look at the relationship between Age & Survived for each Sex.
#Graphical Representation
ggplot(imputedtitanicreengineered,aes(x = Age, fill = factor(Survived))) + geom_histogram()+ facet_grid(.~Sex)
#Table
table(imputedtitanicreengineered$Sex,imputedtitanicreengineered$Survived)
#Result:It is crystal clear from the above graph that distribution histogram curve has identical pattern age wise for both Survived as well as Non-Survived people; however the peaks are different. 
#Also,it may be noted that female survival % > male survival %.

#Let's now bifurcate Child and Adult
#Graphical Representation
imputedtitanicreengineered$Child[imputedtitanicreengineered$Age < 18] <- 'Child'
imputedtitanicreengineered$Child[imputedtitanicreengineered$Age>= 18] <- 'Adult'
ggplot(imputedtitanicreengineered,aes(x = Age, fill = factor(Survived))) + geom_bar(stat = "count") +facet_grid(.~Child)
#Table
table(imputedtitanicreengineered$Child,imputedtitanicreengineered$Survived)
#Result:It can be statistically derived from the above table and graph that about 50% children on board survived while about 35% adults survived on board.

#Let's further bifurcate Adults into Men, Mother, Not Mother(presumably Miss, if I dare say so) and Child. 
#In order to do that, we will first catagorize 'Mother' & 'Not Mother'. Since the 'Child' is already created, the remaining from the list of 'Age' will be men. 
#Alternatively, it could be found out by tabulating Sex Vs Child.
imputedtitanicreengineered$Mother <- 'Not Mother'
imputedtitanicreengineered$Mother[imputedtitanicreengineered$Sex == 'female'& imputedtitanicreengineered$Age > 18 & imputedtitanicreengineered$Parch > 0 & imputedtitanicreengineered$Salutation != 'Miss'] <- 'Mother'
table(imputedtitanicreengineered$Mother,imputedtitanicreengineered$Survived)
#It can be summarized from the above table that about 71% 'Mothers' on board survived while about 35% 'Not Mothers' survived on board.

#Converting the character variables into factor variables
list1 <- c("Child","Mother")
imputedtitanicreengineered[list1] <- lapply(imputedtitanicreengineered[list1],function(x) as.factor(x))
variables <- c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Salutation","Deck","FamilySize","Child","Mother","Survived")
imputedtitanicreengineered <- imputedtitanicreengineered[,variables]
str(imputedtitanicreengineered)

#Let's see the survival rate wrt Passanger Class.
p1 <- ggplot(imputedtitanicreengineered,aes(x = Pclass,fill = factor(Survived)))+geom_bar(stat = "count", position = "stack")
p2 <- ggplot(imputedtitanicreengineered,aes(x = Pclass,fill = factor(Survived)))+geom_bar(position = "fill")+labs(y = "Proportion")
grid.arrange(p1,p2,ncol=2)
#Result: It can cited from the from the above graphs that about 58% of 1st class passengers, 40% of 2nd class passengers and 27% of 3rd Class passengers survived.

#Survival with respect to Embarkation
#Graph#1
par(mfrow = c(1,2))
qplot(Embarked,data = imputedtitanicreengineered,geom = "density",fill = factor(Survived), alpha = I(0.5),xlab = "Embarked", ylab = "Density", main = "Kernel Density Plot for Port of Embarkment And Survival")
#Graph#2
ggplot(imputedtitanicreengineered,aes(x = Embarked, fill = factor(Survived))) + geom_bar(position = "fill")+labs(y="Proportion")
#Result:Kernel Density Plot shows Port "C" is the only port where survival rate is higher than Port "Q" & "S" 
#where non-survival rate is higher than survival. 
#It is evident that about 51% of passengers embarked at "C" port managed to survive 
#while only about 40% & 34% passengers embarked on port "Q" & "S" managed to survive.

#Let us now analyze Salutation of Survivals against the Deck.
ggplot(imputedtitanicreengineered,aes(Survived, fill= factor(Survived))) + geom_bar(stat = "count") + facet_grid(Deck ~Salutation)
#Result:It can deciphered from the above graph that people with 'Miss' and 'Mr' salutations had the highest rate of survival and non-survivals respectively in almost all the decks.

#Now let us analyze the average and highest fare paid for each of the Passenger Class
#Table
tapply(imputedtitanicreengineered$Fare,imputedtitanicreengineered$Pclass,mean)
#Graphical Representation
qplot(Fare,Pclass,data = imputedtitanicreengineered, geom = c("point","smooth"), method = "lm", formula = y~x, col = Sex, main = "Regression of Fare on Passenger Class By Each Sex", ylab = "Passenger class", xlab = "Fare")
#Result:
#Among many conclusions that can be drawn using regression, 
#the easiest will be the fact that the highest passenger fares were not only paid by females for Class 1 & 3 
#but also they were substantially higher than the mean value for the respective class.

#Now let us analyze the average and highest fare paid for each of the Decks
#Table
tapply(imputedtitanicreengineered$Fare,imputedtitanicreengineered$Deck,mean)
#Graphical Representation
qplot(Deck,Fare, data = imputedtitanicreengineered, geom = c("boxplot"), fill = Sex, main = "Fare Per Deck",xlab = "", ylab = "Fare")
#Result:#It can be deduced from the above table & chart that Decks "B" & "C" have the highest fare paid 
#and invariably in almost all decks except deck "T" & the average fare paid by females is higher than males.

#Conditioning Plot for Age Vs Salutations conditioned under Survival status
coplot(Age~Salutation|Survived,data = imputedtitanicreengineered, panel = panel.smooth,xlab = "Salutations",ylab = "Age",columns = 2)
# Result: It can be inferred from the above graph that average age of the survived people with salutations "Mr" and "Miss" is slightly higher than those who did not survive. More or less, 
#there is no difference observed for the average ages for Survived and Not Survived people across salutations.

#Let's plot Violin graph for Passenger Class & Age
ggplot(imputedtitanicreengineered, aes(x=Pclass,y=Age,fill = factor(Survived))) + geom_violin()
#Result: Above graph divulges that people above 60 years of age had a very slim chance of survival in 2nd & 3rd Class in comparison with 1st Class.

#Let's plot that whether the the Women and Children first approach adopted by rescuers on the Titanic?
imputedtitanicreengineered %>%
  filter(Fare <= 300)%>%
  ggplot(mapping = aes(x = Age, y = Fare)) +
  geom_point(aes(colour = Survived, size = Fare, alpha = 0.7)) +
  geom_smooth(se = FALSE)+
  facet_grid(Sex~Pclass, scales = "free") +
  labs(title = "Priority and pattern of rescue on the Titanic",
       x = "Age",
       y = "Fare",
       subtitle = "Children and women in order of ticket class were\nconsidered first in the rescue plan with priority been\nwomen, children and older adults >= 60yrs") +
  theme(
    plot.subtitle = element_text(colour = "#17c5c9",
                                 size=14))+
  theme_bw()
#Result:
# This plotting confirms that the "women and children first" approach was adopted by the rescuers on titanic

# Correlation
#find the correlation
#After inspecting the available features individually you might have realized that some of them are likely to be connected. 
#Does the age-dependent survival change with sex? 
#How are P Class and Fare related?
# Are they strongly enough connected so that one of them is superfluous? Let's find out.
library(corrplot)
mat= cor(titanicreengineered[,c("PassengerId","Survived","Pclass","Age","SibSp","Parch","Fare")], use="complete")
 corrplot(mat)
#Result:1. Survived is correlate most to Sex, and then to Pclass. Fare and Embarked might play a secondary role; the other features are pretty weak
#2.Fare and Pclass are strongly related (1st-class cabins will be more expensive)
#3.A correlation of SibSp and Parch makes intuitive sense (both indicate family size)
#4.Pclass and Age seem related (richer people are on average older? not inconceivable)

#We take this overview plot as a starting point to investigate specific multifeature comparisons in the following.
# Those examinations will likely result in more questions
 
#Correlation between Pclass and Fare
 ggplot(titanicreengineered, aes(Pclass, Fare, colour = Survived)) +
   
   geom_boxplot() +
   
   scale_y_log10()
 #Result:The different Pclass categories are clustered around different average levels of Fare. 
 #This is not very surprising, as 1st class tickets are usually more expensive than 3rd class ones.
 #In 2nd Pclass, and especially in 1st, the median Fare for the Survived == 1 passengers is notably higher than for those who died. 
 #This suggests that there is a sub-division into more/less expensive cabins (i.e. closer/further from the life boats) even within each Pclass.
 
 
 titanicreengineered %>%
   
   ggplot(aes(Fare, fill=Pclass)) +
   
   geom_density(alpha = 0.5) +
   
   scale_x_log10() +
   
   facet_wrap(~ Survived, ncol = 1)
#Result:There is a surprisingly broad distribution between the 1st class passenger fares
# There's an interesting bimodality in the 2nd class cabins and a long tail in the 3rd class ones
# For each class there is strong evidence that the cheaper cabins were worse for survival.

#Correlation between Pclass vs Embarked
#we plot the frequency of the Embarked ports for the different Pclass factors and add a facet to split by the Survived factor:
 titanicreengineered %>%
   
   filter(Embarked %in% c("S","C","Q")) %>%
   
   ggplot() +
   
   geom_bar(aes(Embarked, fill = Pclass), position = "dodge") +
   
   facet_grid(~ Survived)
 #Result:Embarked == Q contains almost exclusively 3rd class passengers
 #The survival chances for 1st class passengers are better for every port. In contrast, the chances for the 2nd class passengers were relatively worse for Embarked == S whereas the frequencies for Embarked == C look comparable.
 #3rd class passengers had bad chances everywhere, but the relative difference for Embarked == S looks particularly strong.
 
 #Correlation between Pclass vs Age
 #We will plot Age vs Fare and facet then by 2 variables, Embarked and Pclass, to create a grid.
 # In addition, we use different colours for the Survived status and different symbols for Sex.
 titanicreengineered %>%
   
   filter(Embarked %in% c("S","C","Q")) %>%
   
   ggplot(mapping = aes(Age, Fare, color = Survived, shape = Sex)) +
   
   geom_point() +
   
   scale_y_log10() +
   
   facet_grid(Pclass ~ Embarked)
 #Result:
 #Pclass == 1 passengers seem indeed on average older than those in 3rd (and maybe 2nd) class. Not many children seemed to have travelled 1st class.
 #Most Pclass == 2 children appear to have survived, regardless of Sex
 #More men than women seem to have travelled 3rd Pclass, whereas for 1st Pclass the ratio looks comparable. Note, that those are only the ones for which we know the Age, which might introduce a systematic bias.

  #Correlation between Age vs Sex
 #Here we are using a density plot with colour overlap and facetting:
 ggplot(titanicreengineered, aes(x=Age)) +
   
   geom_density(aes(fill = Survived), alpha = 0.5) +
   
   facet_wrap(~Sex)
 #Result:
 #Younger boys had a notable survival advantage over male teenagers, whereas the same was not true for girls to nearly the same extent.
 #Most women over 60 survived, whereas for men the high-Age tail of the distribution falls slower.
 
 
 #Correlation between  Parch vs SibSp
 #Next, we will have a closer look at the family relation features Parch (number of parents or children on board) and SibSp (number of siblings or spouses on board). 
 #In order to see how many cases there were for each combination we will use a count plot:
 titanicreengineered %>%
   
   ggplot(aes(Parch, SibSp, color = Survived)) +
   
   geom_count()
 #Result:
 #A large number of passengers were travelling alone.
 #Passengers with the largest number of parents/children had relatively few siblings on board.
 #Survival was better for smaller families, but not for passengers travelling alone.
 
 #Correlation between  Parch vs Sex
 #Another correlation that piqued our interest in the overview plot was the one between Parch vs Sex. 
 #Here we examine it in more detail using a barplot:
 titanicreengineered%>%
   
   ggplot() +
   
   geom_bar(aes(Parch, fill = Sex), position = "dodge") +
   
   scale_y_log10()
 #Result:Many more men traveled without parents or children than women did. The difference might look small here but that's because of the logarithmic y-axis.
 #The log axis helps us to examine the less frequent Parch levels in more detail: Parch == 2,3 still look comparable. 
 #Beyond that, it seems that women were somewhat more likely to travel with more relatives. 

 #Correlation between Age vs SibSp
 #The final correlation we noticed was between the Age and SibSp features. Naively, one would expect that a larger number of siblings would indicate a younger age; 
 #i.e. families with several kids travelling together. (Larger numbers of spouses would be unusual.)
 titanicreengineered %>%
   
   mutate(SibSp = factor(SibSp)) %>%
   
   ggplot(aes(x=Age, color = SibSp)) +
   
   geom_density(size = 1.5)
 #Result:
 #The highest SibSp values (4 and 5) are indeed associated with a narrower distribution peaking at a lower Age. Most likely groups of children from large families.
 #This will lead to a certain degree of interaction between Age and SibSp with respect to the impact on the Survived status. It might also allow us to predict Age from SibSp with a relatively decent accuracy for the higher SibSp values.

 #Confidence Intervals

 #To better understand the confidence we will solve some questions
 #Question 1:make a point estimate for the population mean of the Age column by taking a sample of 40 passengers.
 #Then calculate the difference between your sample mean and the true mean.
 #Solution

 library(dplyr) 
 pop_ages <- titanicreengineered$Age
 pop_ages <- pop_ages[!is.na(pop_ages)]
 
 true_mean_age <- mean(pop_ages)
 cat('True Mean:\t\t', true_mean_age, '\n', sep='')
 
 #we take the sample size less than or equal to 40
 sample_size <- 40
 set.seed(20211020) # Today's date for reproducibility
 
 sample_ages <- sample(pop_ages, size=sample_size)
 sample_mean <- mean(sample_ages)
 
 #Question 2:Calculate the margin of error for the point estimate for a 95% confidence interval. Use the t distribution.
 sample_stdev <- sd(sample_ages)
 t_critical <- qt(0.975, df=(sample_size - 1))
 
 moe <- t_critical * sample_stdev / sqrt(sample_size)  # Margin of Error
 
 cat( moe, sep='')
 
 #Question 3:Calculate and print a 95% confidence interval for the mean of age using the point estimate and margin of error you calculated above. 
 #Make sure that your confidence interval matches the one produced by the call to t.test(age_sample).
 interval = round(c(sample_mean - moe, sample_mean + moe), 1)
 cat("Manual t-test, 95% Confidence Interval:\n\t",
     interval[1],
     " - ",
     interval[2],
     sep='')
 #Making sure that the confidence interval matches
 built_in <- t.test(sample_ages)$conf.int[1:2]%>%
   format(digits=3) %>%
   paste(collapse=' - ')
 cat("Built-in t-test, 95% Confidence Interval:\n\t", built_in, sep = '')
 cat('Point Estimate Mean:\t', sample_mean, '\n', sep='')
 cat('Mean Difference:\t', true_mean_age - sample_mean, '\n', sep='')
 
 #Hypothesis Testing
 #Question1:Using t-Test, we will try to check the Null Hypothesis (H0) for Fare and Survivals.
 
 # H0 : There is no difference between the two population means.
 # H1 : There is difference between the two population means.
 t.test(Fare~Survived,data = imputedtitanicreengineered)
 #Result:As the p-value <0.05, we reject H0 i.e. there is difference between the two population means. 
 #Further, average fare paid by those who survived is higher by $25 than those who did not survive.
 
 #Question 2:Using t-Test, we will try to check the Null Hypothesis (H0) for Age and Survivals.
 
 # H0 : There is no difference between the two population means.
 # H1 : There is difference between the two population means.
 
 t.test(Age~Survived,data = imputedtitanicreengineered)
 #Result:As the p-value is relatively higher but less than 0.05, we can tentatively say that we reject the H0 i.e. there is a minor difference between the two population means. 
 #Further, the same is proved by the values of means i.e. average age of those who did not survive is slightly higher by 2 years than those who managed to survive.
 
 #Question3:Using Chi-Square Test, we will try to check the Null Hypothesis (H0) for the Passenger Class and Survivals.
 # H0 : Passenger Class and Survivals are independent.
 # H1 : Passenger Class and Survivals are not independent.
 chisq.test(imputedtitanicreengineered$Pclass,imputedtitanicreengineered$Survived)
 
 #Result:As the p-value <0.05, we reject H0 i.e. Passenger Class and Survivals are not independent. 
 #This conclusion matches with our exploratory analysis as shown earlier.
 
 #Question4:run a t-test to test the following hypothesis: H2: The Titanic survivors were younger than the passengers who died.
 #H0 : The Titanic survivors weren't younger than the passengers who died.
 #H1 : The Titanic survivors were younger than the passengers who died.
 
 aggregate(Age~Survived, data=titanicreengineered, mean)
 t.test(Age ~ Survived, data=titanicreengineered)
 
 #Result:This is less than 0.05(the standard value), so we reject this Null Hypothesis.
 
 #Question 5: Run a Pearson's Chi-squared test to test that The proportion of females on-board who survived the sinking of the Titanic was higher 
 #than the proportion of males on-board who survived the sinking of the Titanic.
 
 #H0 : The proportion of females on-board who survived the sinking of the Titanic was higher 
 #than the proportion of males on-board who survived the sinking of the Titanic.
 
 #H1 : The proportion of females on-board who survived the sinking of the Titanic was lower 
 #than the proportion of males on-board who survived the sinking of the Titanic. 
 
 library(psych)
 describe(titanicreengineered$Age)
 mytable <- with(titanicreengineered, table(Survived))
 mytable  #where 1= survival
 prop.table(mytable)*100 #where 1= survival
 mytable <- xtabs(~ Survived+Pclass, data=titanicreengineered)
 mytable # here 1=survival , and for class 1= ist class
 margin.table(mytable)  #to get the total no of passengers
 prop.table(mytable)*100 #to get the percentage of first class passenger who survived to total no of passenger
 mytable <- xtabs(~ Survived+Sex+Pclass, data=titanicreengineered)
 ftable(mytable) #here 1=survival , and for class 1= First class
 mytable <- xtabs(~ Survived+Sex, data=titanicreengineered)
 margin.table(mytable)  #for reference for total no of passengers
 prop.table(mytable)*100
 margin.table(mytable, 2)
 prop.table(mytable, 2) #where 1= survived
 mytable <- xtabs(~ Sex+Survived, data=titanicreengineered)
 addmargins(mytable) #to just have a guess or to review
 chisq.test(mytable)

 #Result:as after running the Chi-squared test
 #we can say(p < 0.01) which means the null hypothesis is correct.
 
 #Analysis Of Variance
 numerical_variable ~ categorical_variable. 
 Age ~ Pclass.
 titanicANOVA <- lm(Age ~ Pclass, data = titanicreengineered)
 anova(titanicANOVA)
 #Result:
 #This table shows the results of a test of the null hypothesis that the mean ages are the same among the three groups. The P-value is very small, 
 #and so we reject the null hypothesis of no differences in mean age among the passenger classes.
 #Tukey-Kramer test
 #A Tukey-Kramer test lets us test the null hypothesis of no difference between the population means for all pairs of groups. The Tukey-Kramer test (also known as a Tukey Honest Significance Test, or Tukey HSD), 
 #is implemented in R in the function TukeyHSD().
 
 TukeyHSD(aov(titanicANOVA))
 #Result:
 #In the case of the Titanic data, P is less than 0.05 in all pairs, and we therefore reject every null hypothesis.
 # We conclude that the population mean ages of all passenger classes are significantly different from each other.
 
 #Kruskal-Wallis test
 #A Kruskal-Wallis test is a non-parametric analog of a one-way ANOVA. It does not assume that the variable has a normal distribution.
 # (Instead, it tests whether the variable has the same distribution with the same mean in each group.)
 kruskal.test(Age ~ Pclass, data = titanicreengineered)
 #Result:
 #You can see for the output that a Kruskal-Wallis test also strongly rejects the null hypothesis of equality of age for all passenger class groups with the Titanic data.
 
 ## Linear and Multiple Regression Models:
 #Building Models
 
 trainingdataset <- imputedtitanicreengineered[c(1:891),]
 testingdataset <- imputedtitanicreengineered[c(892:1309),]
 
 #Model 1: Linear Regression:
 trainingdataset$model <- rep(0,891)
 nrow(trainingdataset[trainingdataset$model==
              train$Survived,])/891 
 #Result:If we predict that everyone dies, we're right 61.6 % of the time in the training set.
 
 #A Gender Model
 (gender_model <- lm(Survived~Sex, data=trainingdataset))
 
 #Result: This model predicts that females have a 74.2% of surival and males have a 74.2 - 55.3 = 18.9% chance of survival.
 
 #Predictions with the Gender Model
 trainingdataset$gender_model_pred <- round(predict(gender_model, train))
 #We can make "in sample" predictions (meaning, on the training set) with this model and round them to make them 0/1 predictions rather than predicting probabilities.
 
 #In Sample Accuracy
 nrow(trainingdataset[trainingdataset$gender_model_pred==
              trainingdataset$Survived,])/891
 #Result: In Sample, we're now right 78.7 % of the time but this isn't the true test. Models with additional variables will always do better in sample.
 
 #MODEL 2 : Random Forest
 model2 <- randomForest(Survived ~ ., data = trainingdataset, importance = TRUE)
 model2
 #Random Forest model shows an Out Of Sample Error Rate of about 17%
 varImpPlot(model2)

 
 #MODEL 3 : Support Vector Machine
 model3 <- svm(Survived ~ ., data = trainingdataset)
 model3
 
 
 #=============================================THANK YOU=======================================================================#
 

 
 
 
 

 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
