#------------------------------Projet MA--------------------------------------#
#-------------------------khalil al sayed---------------------------------------#
#-----------------------------ISN---------------------------------------------#
#-------------------------Analyse descriptive-------------------------------------------#
getwd()
setwd("C:/Users/Amina/Documents/zahira R")
library(readxl)
library(dplyr)
library(ggplot2) 
library(data.table) #pour utiliser la fonction setnames ou rename
data<-read_excel("default.xls")

#-------------------Verifier s'il n'ya pas de valeur manquante-------------# 
sum(as.numeric(is.na(data)))
#----------------------------------------------------------------------------

#On transfore la variable Y qui represente le groupe � pr�dire ici
#outcome en variable de facteur 
#data$Y<-as.factor(data$Y)
#----------------------------------------------------------------------------
#-------------------------On regarde un apercu du data----------------------- 
head(data)
#-----------------------------------------------------------------------------

#Pour simplifier la claritée et la compréhension des données 
#on renome quelques variables selon leurs significations:
#--------------------On renomme nos variables--------------------------------- 
data=rename(data, montant.credit=X1)
data=rename(data, genre=X2)
data=rename(data, education=X3)
data=rename(data, situation.familiale=X4)
data=rename(data, age=X5)
data=rename(data, defaut.payement=Y)
#------------------------------------------------------------------------------
#----------------------------On �tudie de la correlation-----------------------
#------------------------etude sur les variables numérique:------------------
M=data
M=cor(data)
library(corrplot)
corrplot(M)
#-----------------------------------------------------------------------------
#Analysons ces données on faisant un summary et un str-----------------------
summary(data)
str(data)
#-----------------------------------------------------------------------------
#----------------transformation de chaque variables concernée----------------
genre=as.factor(data$genre)
education=as.factor(data$education)
situation.familiale=as.factor(data$situation.familiale)
age=as.factor(data$age)
X6=as.factor(data$X6)
X7=as.factor(data$X7)
X8=as.factor(data$X8)
X9=as.factor(data$X9)
X10=as.factor(data$X10)
X11=as.factor(data$X11)
defaut.payment=as.factor(data$defaut.payement)
data=data[, -c(2:3)]
data=data[,-2]
data=data[,-c(3:8)]
data=data[,-15]
DATA=cbind(genre,education,situation.familiale,X6,X7,X8,X9,X10,X11,data,defaut.payment)
DATA=as.data.frame(DATA)#24 car y le vecteur qui distingues les lingnes numeroté de 1 .. 30000

#---visualisation
pie(table(DATA$genre))
plot(education, col="purple")
plot(situation.familiale, col="pink")
plot(age, col="gold")

#------------------------------------------------------------------------------
#Donc aprés toutes ces transformations faisons un sumarry et un str
#pour verifier si les variables sont bien representées:\
summary(DATA)
str(DATA)
#------------------------------------------------------------------------------
#Nous allons effectuer quelques tests sur les differentes variables:
# Variables qualitatives: par exemple age, genre ,
#education, situation familiale et la varible Y: default payment.\

#Création des tableaux croisés:
#------ variable genre:
table_genre<-table(DATA$defaut.payment,DATA$genre)
rownames(table_genre)<-c("defaut de payment"," pas de defaut de payment")
colnames(table_genre)<-c("homme","femme")
table_genre

#-----variable education
table_education<-table(DATA$defaut.payment,DATA$education)
rownames(table_education)<-c("defaut de payment"," pas de defaut de payment")
#colnames(table_education)<-c("secandaire","université","supérieur", #"autre","autre","autre") #ya un pb index jusqua 6 au lieu de 3
table_education

#----- variable situation:
table_situation<-table(DATA$defaut.payment,DATA$situation.familiale)
rownames(table_situation)<-c("defaut de payment"," pas de defaut de payment")
#colnames(table_situation)<-c("marié","celibataire","autre")
table_situation

#---visualisation
barplot(table_genre,beside=TRUE,col = c("red","green"))
legend(x="topleft",legend=c("defaut de payment"," pas de defaut de payment"), fill=c("red","green"))
barplot(table_education,beside=TRUE, col=c("red","green"))
legend(x="topright",legend=c("defaut de payment"," pas de defaut de payment"), fill=c("red","green"))
barplot(table_situation,beside=TRUE,  col=c("red","green"))
legend(x="right",legend=c("defaut de payment"," pas de defaut de payment"), fill=c("red","green"))
ggplot(DATA, aes(montant.credit,age, color= genre))+
geom_point(size=2)+ggtitle("genre: montant.credit vs age")
ggplot(DATA, aes(montant.credit,situation.familiale, color=X12))+
geom_point(size=2)+ggtitle("X12: montant.credit vs situation.familiale")
ggplot(DATA, aes(X18,education, color=X7))+
  geom_point(size=2)+ggtitle("X7: montant.credit vs situation.familiale")


#---tests:
chisq.test(table_genre) #p-value = 4.945e-12
chisq.test(table_education) # p-value < 2.2e-16
chisq.test(table_situation) # p-value = 8.826e-08


















