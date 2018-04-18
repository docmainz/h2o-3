setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../scripts/h2o-r-test-setup.R")

# test derived from Nidhi/customer code
test.pub.5456 <- function() {

  N=10
  set.seed(5)
  ID = sample(c(12,13,14,15,16,17),size=N,replace=TRUE)
  num = rnorm(N,mean = 12,sd = 21212)
  Group_by_column = sample(c(1,2,3,4,5),size=N,replace=TRUE) #
  data = sample(c(0,1),size = N,replace = T)

  dd = data.frame(ID,Group_by_column,num,data)
  data1 = as.h2o(dd)

  N=10
  set.seed(20)
  ID = sample(c(12,13,14,15,16,17),size=N,replace=TRUE)
  num_1 = rnorm(N,mean = 12,sd = 21212)
  Column_to_arrange_by = sample(c(1,2,3,4,5),size=N,replace=TRUE) #
  data_1 = sample(c(0,1),size = N,replace = T)
  fdata = factor(data_1)
  table(fdata)
  dd2 = data.frame(ID,Column_to_arrange_by,num_1,fdata)
  data2 = as.h2o(dd2)

  tt = h2o.merge(data1,data2,by = "ID")  #group by id
  #as.data.frame(tt)

  rr = h2o.arrange(x = tt,"ID","Group_by_column")  #sort by rank

  #================Modified================
  h2o.head(tt)
  dt_tt<- as.data.table(tt)

  #Arrange by group and mutate a new rank column
  library(dplyr)
  library(data.table)

  #Sort the input data table - important to get accurate results
  dt_tt1 <- arrange(dt_tt,Column_to_arrange_by)
  dt_tt_RANK <- dt_tt1 %>%
    group_by(Group_by_column) %>%
    mutate(New_Rank_column = order(Column_to_arrange_by, decreasing=FALSE))%>%as.data.frame()
  head(dt_tt_RANK)
  #Import into H2o
  tt1.HEX <- as.h2o(dt_tt1)

  #Max aggregation by group and store it in a new column
  max_function <- function(df) { max(df[,1], na.rm = TRUE) }
  tt1_MAX.HEX <- h2o.ddply(tt1.HEX, "Group_by_column", max_function)
  h2o.head(tt1_MAX.HEX)

  #Rank and ddply *****
  rank_function <- function(df) { h2o.arrange(df, df[,3]) }
  tt1_RANK.HEX <- h2o.ddply(tt1.HEX, "Group_by_column", rank_function)
}

doTest("PUBDEV-5456-rank-within-group", test.pub.5456)
