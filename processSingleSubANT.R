## Processing pipeline for single subject ANT data


processSingleSubANT <- function(dataFile) {
  
  ##TODO##
  ## add header information that says all 
  ## inputs and outputs of function
  ### inputs
  ##### dataFile <-  this is the file which you will be cleaning
  ##### df <-  this is the dataframe that will be created once you read in the to be cleaned .txt file
  ##### df.ant2 <-  is the dataframe which has removed all unnecessary variables from the original dataframe
  ### outputs
  ##### outputname <-  this will be the file name that the user enters

  
  
  #dependencies
  library(tidyverse)
  library(tools)
  
  ##TODO##
  ## Put in if-statement checking nRows on the iout file
  ## if nRows = 300, its pre and first 12 should be dropped
  ## if nRows = 288 its post and no dropping is needed
  ## else give warning that file size does not fit expectations
  #####

  # read data
  df <- read.csv(dataFile, fileEncoding='UCS-2LE')
  
  
  ##TODO## 
  # Test this
  if (nrow(df) == 300) {
    df$pre_or_post <- 'pre'
    df <- df[-c(1:12),]
    print('
          ')
    print("First 12 rows of data are being removed as they are just practice")
  } else if (nrow(df) == 288) {
    df$pre_or_post <- 'post'
    print('
          ')
    print('This is post-TMS session - no dropped rows')
  } else {
    message('Unexpected number of rows in file')
    return()
  }
  
  # # select relevant columns
  # df.ant2 <- df %>%
  #   dplyr::select(Subject, pre_or_post, DataFile.Basename, Session, SessionDate,
  #                 SessionTime, Block, Trial, DurationOfFixation,
  #                 FlankerType, WarningType, SlideTarget.RT,
  #                 SlideTarget.RESP, SlideTarget.CRESP,
  #                 SlideTarget.ACC)

 
 
  # compute median RTs and nTrials per FlankerType and WarningType condition
  medRT_condition <- df %>%
    dplyr::filter(SlideTarget.RT > 200 & SlideTarget.RT < 1200, SlideTarget.ACC == 1) %>%
    dplyr::group_by(FlankerType, WarningType) %>%
    dplyr::summarise(medRT = median(SlideTarget.RT),
                     nTrials = n()) %>%
    tidyr::unite(condition, FlankerType, WarningType) %>%
    tidyr::gather(variable, value, medRT:nTrials) %>%
    tidyr::unite(temp, condition, variable) %>%
    tidyr::spread(temp, value) 
  
  # compute median RTs and nTrials per FlankerType condition
  medRT_flankerType <- df %>%
    dplyr::filter(SlideTarget.RT > 200 & SlideTarget.RT < 1200, SlideTarget.ACC == 1) %>%
    dplyr::group_by(FlankerType) %>%
    dplyr::summarise(medRT = median(SlideTarget.RT),
                     nTrials = n()) %>%
    tidyr::gather(variable, value, medRT:nTrials) %>%
    unite(temp, FlankerType, variable) %>%
    tidyr::spread(temp, value)
  
  # compute median RTs and nTrials per WarningType condition
  medRT_warningType <- df %>%
    dplyr::filter(SlideTarget.RT > 200 & SlideTarget.RT < 1200, SlideTarget.ACC == 1) %>%
    dplyr::group_by(WarningType) %>%
    dplyr::summarise(medRT = median(SlideTarget.RT),
                     nTrials = n()) %>%
    tidyr::gather(variable, value, medRT:nTrials) %>%
    unite(temp, WarningType, variable) %>%
    tidyr::spread(temp, value)
  
  # compute median RTs and nTrials over whole all trials
  med_vec_RT_allcond <- df %>%
    dplyr::filter(SlideTarget.RT > 200 & SlideTarget.RT < 1200, SlideTarget.ACC == 1) %>%
    dplyr::summarise(medRT = median(SlideTarget.RT),
                     nTrials = n())
  
  
  print('computing your cues and congruency variables...')
  
  
  # bind computed median RTs into single row vector 
  medRT.row <- bind_cols(medRT_condition, medRT_flankerType, medRT_warningType, med_vec_RT_allcond)
  return(medRT.row)
  
}
















