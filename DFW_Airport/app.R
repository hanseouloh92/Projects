## app developed by mhan


library(shiny)
library(shinythemes)
library(readxl)
library(dplyr)
library(tidyverse)
library(stringr)
library(ggplot2)
library(lubridate)
library(scales)
library(plotly)
library(tidytext)
library(tidyr)
library(stringr)
#install.packages('rpsychi') used to do one way ANOVA comparison based on summary statistics
library(rpsychi)
library(stringr)
#install.packages('DT')
library(DT)

filePath = 'data/Inspections Export 512019.xlsx' ## filePath where the excel is located --- should be changed to direct towards DBMS 
mainDF <- read_excel(filePath)
names(mainDF) <- make.names(colnames(mainDF), unique=TRUE) # format column names for programming ease


############### data cleaning and formatting
mainDF = mainDF[!is.na(mainDF$Scale.Type), ]
mainDF = mainDF[!is.na(mainDF$Item.Rating),]
mainDF = mainDF[!(mainDF$Ticket.Type...6 == 'Manual Ticket'),]
mainDF$Ticket.Status[is.na(mainDF$Ticket.Status)] <- 'Completed' # NAs in ticket status I classified as "completed" -- dont know difference between open and closed tickets
mainDF$SCORE <- ifelse((mainDF$Item.Rating >=4 & mainDF$Scale.Type == 'Custodial') |(mainDF$Item.Rating >2 & mainDF$Scale.Type == 'Maintenance'),
                       'PASS', "FAIL") # at the time of programming scores were 0,1,4 for custodial and 0,1,3 for maintance. assigned "binary" classification based on this
mainDF$DATE <- as.Date(mainDF$Inspection.Entered.On, "%d-%b-%Y") # date formmatting



ui <- fluidPage(
  tabsetPanel(
    tabPanel('Cleanliness and Maintenance Stats',
             titlePanel('Cleanliness & Maintenance Scores'),
             sidebarLayout(
               sidebarPanel(
                 dateRangeInput('dateRange', 'Date Range:',
                                start = sort(mainDF$DATE)[1],
                                end = tail(sort(mainDF$DATE), n = 1),
                                min = sort(mainDF$DATE)[1],
                                max = tail(sort(mainDF$DATE), n = 1)),
                 selectInput('Terminal', 'Select a Terminal',
                             c('A', 'B', 'C', 'D', 'E'),
                             selected = 'A'),
                 selectInput('Contract', 'Select Performance',
                             c('Restroom', 'Gate Lounge', 'Overall'),
                             selected = "Restroom"),
                 helpText('Select Terminal and Performance, then push Update'),
                 submitButton('Update')
               ),
               mainPanel(plotOutput(outputId = 'barplot'),
                         fluidRow(column(width = 6, h2('Highest Performing Custodial Areas')), column(width = 6, h2('Lowest Performing Custodial Areas'))),
                         fluidRow(column(width = 6, tableOutput(outputId = 'TopTable')), column(width = 6, tableOutput(outputId = 'LowTable')))
               )
             )
    ),
    ########################################################################################################################################################################                    
    tabPanel('Terminal Comparison',
             titlePanel('Terminal Comparison'),
             sidebarLayout(
               sidebarPanel(
                 dateRangeInput('dateRange1', 'Date Range:',
                                start = sort(mainDF$DATE)[1],
                                end = tail(sort(mainDF$DATE), n = 1),
                                min = sort(mainDF$DATE)[1],
                                max = tail(sort(mainDF$DATE), n = 1)),
                 selectInput('Terminal1', 'Select a Terminal',
                             c('A', 'B', 'C', 'D', 'E'),
                             selected = 'A'),
                 selectInput('Contract1', 'Select Performance',
                             c('Restroom', 'Gate Lounge', 'Overall'),
                             selected = "Restroom"),
                 helpText('Select Terminal and Performance, then push Update'),
                 submitButton('Update')  
               ),
               mainPanel(
                 fluidRow(column(width = 6, h3('Terminal Custodial Performance Comparison')), column(width = 4, h3('Terminal Custodial Rankings'))),
                 fluidRow(column(width = 6, dataTableOutput(outputId = 'ComparisonCust')), column(width = 4, dataTableOutput(outputId = 'ranking'))),
                 fluidRow(column(width = 5, htmlOutput('titleTab2')))
               )
             )
    ),
    ##########################################################################################################################################################################
    tabPanel('Inspection Ticket Stats',
             titlePanel('Inspection Performance'),
             sidebarLayout(
               sidebarPanel(
                 dateRangeInput('dateRange2', 'Date Range:',
                                start = sort(mainDF$DATE)[1],
                                end = tail(sort(mainDF$DATE), n = 1),
                                min = sort(mainDF$DATE)[1],
                                max = tail(sort(mainDF$DATE), n = 1)),
                 selectInput('Terminal2', 'Select a Terminal',
                             c('A', 'B', 'C', 'D', 'E'),
                             selected = 'A'),
                 helpText('Select Terminal and Date Range to update the table to the right'),
                 submitButton('Update'),
                 htmlOutput('total')
               ),
               mainPanel(h2('# of Tickets Open/Closed for Employee'),
                         textOutput('Caption'),
                         htmlOutput('terminal_name'),
                         dataTableOutput(outputId = 'ticketTable')
               )
             )
    ),
    ##########################################################################################################################################################################
    tabPanel('Sentiment Analysis',
             titlePanel('Sentiment Analysis'),
             sidebarLayout(
               sidebarPanel(
                 dateRangeInput('dateRange3', 'Date Range:',
                                start = sort(mainDF$DATE)[1],
                                end = tail(sort(mainDF$DATE), n = 1),
                                min = sort(mainDF$DATE)[1],
                                max = tail(sort(mainDF$DATE), n = 1)),
                 selectInput('Terminal3', 'Select a Terminal',
                             c('A', 'B', 'C', 'D', 'E'),
                             selected = 'A'),
                 helpText('Select a terminal to view comment sentiment classification'),
                 submitButton('Update')
               ),
               mainPanel(plotOutput(outputId = 'sentimentPlot'),
                         plotOutput(outputId = 'sentimentTime')
               )
             )
    )
  )
)





server <- function(input, output) {
  
  
  terminal <- reactive({input$Terminal})
  terminal1 <- reactive({input$Terminal1})
  terminal2 <- reactive({input$Terminal2})
  terminal3 <- reactive({input$Terminal3})
  contract <- reactive({input$Contract})
  contract1 <- reactive({input$Contract1})
  
  output$barplot <- renderPlot({ # bar plot of pass fails based on terminal and "contract"
    terminalName = paste("Terminal", as.character(terminal()))
    if(contract() == 'Restroom' | contract() == 'Gate Lounge'){
      mainDF %>% filter(Inspection.Entered.By %in% imp.names) %>%
        filter(Terminal.Name ==  terminalName & Inspection.Type.Name == contract() & 
                 (DATE <= input$dateRange[2] & DATE >= input$dateRange[1])) %>%
        group_by(Scale.Type, SCORE) %>% 
        summarise(TOTAL = n()) %>%
        ggplot(., aes(x = Scale.Type, y = TOTAL, fill = factor(SCORE))) +
        geom_bar(position = 'dodge2', stat='identity')+
        labs(title = paste('Cleanliness & Maintenance Scores -',terminalName, str_to_upper(contract())),
             x = 'Ticketing Type',
             y = 'Count',
             fill = 'Scoring')
    }else if(contract() == 'Overall'){ ## overall ignores the contrat classiciation for a broad comparison
      mainDF %>% filter(Inspection.Entered.By %in% imp.names) %>%
        filter(Terminal.Name ==  terminalName & (Inspection.Type.Name == 'Restroom' | Inspection.Type.Name == 'Gate Lounge') &
                 (DATE <= input$dateRange[2] & DATE >= input$dateRange[1])) %>%
        group_by(Scale.Type, SCORE) %>% 
        summarise(TOTAL = n()) %>%
        ggplot(., aes(x = Scale.Type, y = TOTAL, fill = factor(SCORE))) +
        geom_bar(position = 'dodge2', stat='identity')+
        labs(title = paste('Cleanliness & Maintenance Scores -',terminalName, contract()),
             x = 'Ticketing Type',
             y = 'Count',
             fill = 'Scoring')
      # }else if(contract() == 'Friendly Staff'){ ## friendly staff metric was found in the Item.Name with category "Professional Courtesy" 
      #     mainDF %>% 
      #         filter(Terminal.Name ==  terminalName & Item.Name == 'Professional Courtesy' &
      #                    (DATE <= input$dateRange[2] & DATE >= input$dateRange[1])) %>%
      #         group_by(Scale.Type, SCORE) %>% 
      #         summarise(TOTAL = n()) %>%
      #         ggplot(., aes(x = Scale.Type, y = TOTAL, fill = factor(SCORE))) +
      #         geom_bar(position = 'dodge2', stat='identity')+
      #         labs(title = paste('Cleanliness & Maintenance Scores -',terminalName, contract()),
      #              x = 'Ticketing Type',
      #              y = 'Count',
      #              fill = 'Scoring')
    }
  })
  
  # output$pieChart <- renderPlot({ ## pie chart to showing how many open/closed tickets grouped by employee submission
  #     terminalName = paste('Terminal', as.character(terminal1()))
  #     mainDF %>% 
  #         filter(Terminal.Name == terminalName) %>% 
  #         group_by(Ticket.Status) %>%
  #         summarise(TOTAL = n()) %>%
  #         ggplot(., aes(x = "", y = TOTAL, fill = Ticket.Status)) +
  #         geom_bar(width = 1, stat = 'identity') + 
  #         coord_polar('y', start = 0) +
  #         geom_text(aes(label = paste0(round(TOTAL/sum(TOTAL) * 100), '%')), position = position_stack(vjust = 0.5)) + # adding text overlay -- could use cleaning up so the numbers are more readible
  #         labs(x = NULL, y = NULL, fill = NULL, title = paste(terminalName,  "Inspections")) + theme_classic() + 
  #         theme(axis.line = element_blank(), axis.text = element_blank(), axis.ticks = element_blank()) +
  #         scale_fill_manual(values = c('Completed' = 'springgreen3', 
  #                                      'Closed' = 'tomato1', 
  #                                      'Open' = 'lightgoldenrod1'))
  # })
  
  output$TopTable <- renderTable({ # shows the top performaing areas based on the number of "PASS" 
    if (contract() == 'Restroom' | contract() == 'Gate Lounge'){
      mainDF %>% filter(Inspection.Entered.By %in% imp.names) %>%
        filter(Terminal.Name == paste('Terminal', as.character(terminal())) & (Scale.Type == 'Custodial') &
                 (DATE <= input$dateRange[2] & DATE >= input$dateRange[1])) %>% 
        filter(SCORE == 'PASS' & Inspection.Type.Name == contract()) %>% group_by(Asset.Name) %>% 
        summarise(TOTAL= n()) %>% arrange(desc(TOTAL)) %>% head(n = 3) %>% 
        rename('# Total Passed Inspections' = TOTAL, 'Asset Name' = Asset.Name) %>%
        as.data.frame()
    } else if (contract() == 'Overall'){ ## overall includes the a column stating whether the area is a bathroom, gate lounge or whatever it is
      mainDF %>% filter(Inspection.Entered.By %in% imp.names) %>%
        filter(Terminal.Name == paste('Terminal', as.character(terminal())) & (Scale.Type == 'Custodial') & (Inspection.Type.Name == 'Restroom' | Inspection.Type.Name == 'Gate Lounge') &
                 (DATE <= input$dateRange[2] & DATE >= input$dateRange[1])) %>% 
        filter(SCORE == 'PASS') %>% group_by(Inspection.Type.Name, Asset.Name) %>% 
        summarise(TOTAL= n()) %>% arrange(desc(TOTAL)) %>% head(n = 3) %>% 
        rename('# Total Passed Inspections' = TOTAL, 'Asset Name' = Asset.Name, 'Inspection Type' = Inspection.Type.Name) %>%
        as.data.frame()
      # } else if (contract() == 'Friendly Staff'){
      #     mainDF %>%
      #         filter(Terminal.Name == paste('Terminal', as.character(terminal())) & (Scale.Type == 'Custodial') &
      #                    (DATE <= input$dateRange[2] & DATE >= input$dateRange[1])) %>% 
      #         filter(SCORE == 'PASS' & Item.Name == 'Professional Courtesy') %>% group_by(Inspection.Type.Name, Asset.Name) %>% 
      #         summarise(TOTAL= n()) %>% arrange(desc(TOTAL)) %>% head(n = 3) %>% as.data.frame()
    }
    ## change total column name to represent number
  })
  
  output$LowTable <- renderTable({
    if (contract() == 'Restroom' | contract() == 'Gate Lounge'){
      mainDF %>% filter(Inspection.Entered.By %in% imp.names) %>%
        filter(Terminal.Name == paste('Terminal', as.character(terminal())) & (Scale.Type == 'Custodial') &
                 (DATE <= input$dateRange[2] & DATE >= input$dateRange[1])) %>% 
        filter(SCORE == 'PASS' & Inspection.Type.Name == contract()) %>% group_by(Asset.Name) %>% 
        summarise(TOTAL= n()) %>% arrange(TOTAL) %>% head(n = 3) %>% 
        rename('# Total Passed Inspections' = TOTAL, 'Asset Name' = Asset.Name) %>%
        as.data.frame()
      
    } else if (contract() == 'Overall'){ ## overall includes the a column stating whether the area is a bathroom, gate lounge or whatever it is
      mainDF %>% filter(Inspection.Entered.By %in% imp.names) %>%
        filter(Terminal.Name == paste('Terminal', as.character(terminal())) & (Scale.Type == 'Custodial') & (Inspection.Type.Name == 'Restroom' | Inspection.Type.Name == 'Gate Lounge') &
                 (DATE <= input$dateRange[2] & DATE >= input$dateRange[1])) %>% 
        filter((SCORE == 'PASS') & (Inspection.Type.Name != 'Report an Issue')) %>% group_by(Inspection.Type.Name, Asset.Name) %>% 
        summarise(TOTAL= n()) %>% arrange(TOTAL) %>% head(n = 3) %>% 
        rename('# Total Passed Inspections' = TOTAL, 'Asset Name' = Asset.Name, 'Inspection Type' = Inspection.Type.Name) %>%
        as.data.frame()
    }
  })
  
  output$total <- renderUI({ # text output that shows the total number of inspection tickets by terminal 
    ## inspection id represents individual inspections group by 
    ## count pass and fails based on the inspection by name
    dataframe <- mainDF %>% 
      filter(Terminal.Name == paste('Terminal', as.character(terminal2())) & 
               (DATE <= input$dateRange2[2] & DATE >= input$dateRange2[1])) %>% 
      group_by(Inspection.Entered.By, Ticket.Status) %>%
      summarise(TOTAL = n()) %>% as.data.frame()
    htmlTotal <- paste( "<b>",sum(dataframe$TOTAL), "<b>") ## html used to bold the total number of tickets open by terminal
    HTML(paste("</p>",'Total Inspections for',paste('Terminal', as.character(terminal2()), ':', htmlTotal )))
  }) # print total number of inspections
  
  output$terminal_name <- renderUI({
    HTML(paste("<b>",'Terminal', as.character(terminal2()),"Tickets", "<b>"))
  })
  
  output$ticketTable <- renderDataTable({ # table that shows the top 10 employees with the most open/closed tickets
    terminalName = paste('Terminal', as.character(terminal2()))
    mainDF %>% 
      filter((Terminal.Name == terminalName) & 
               (DATE <= input$dateRange2[2] & DATE >= input$dateRange2[1])) %>% 
      mutate(Inspection.ID = as.character(Inspection.ID)) %>%
      group_by(Inspection.ID, Inspection.Entered.By, SCORE) %>% 
      tally() %>% 
      spread(SCORE, n) %>% 
      replace(is.na(.), 0) %>% 
      mutate(TOTAL = PASS+FAIL) %>%
      rename("Inspection Entered By" = Inspection.Entered.By,
             '# Failed Ticks' = FAIL, '# Pass Ticks' = PASS,
             'Inspection Unique ID' = Inspection.ID) %>%
      as.data.frame()
  }, options = list(columnDefs = list(list(targets = seq(2,4), searchable = FALSE)), 
                    dom = 'tp', pageLength = 10, order = list(list(4, 'desc'))),
  rownames = FALSE
  )
  
  output$Caption <- renderText({
    "Search by Inspection ID or Employee for Specifics"
  })
  
  output$titleTab2 <- renderUI({
    HTML(paste("<i>","Grey rows indicate no significant difference between terminals.", "</i>", "<br/>",
               "<b>", 'Terminal:', as.character(terminal1()),"<br/>", 
               'Performance:', contract1(), "</b>"
    ))
  })
  
  output$ComparisonCust <- renderDataTable({ ## performs ANOVA comparison within custodial to determine how the termial performs against other terminals
    terminalName = paste("Terminal", as.character(terminal1()))
    
    # same logic used throughout app -- restroom, gate lounge is defined in the Inspection.Type.Name. "Friendly Staff" is from the Item.Name column
    ## each dpylr creates the expected value column to calculate the mean of the a discrete "random" variable. -- could be incorrect and could use some more analysis 
    if(contract1() == 'Restroom' | contract1() == 'Gate Lounge'){
      df <- mainDF %>% filter(Scale.Type == 'Custodial' &  Inspection.Type.Name == contract1() &
                                (DATE <= input$dateRange1[2] & DATE >= input$dateRange1[1])) %>%
        group_by(Terminal.Name,  Item.Rating) %>% summarise(COUNT = n()) %>% mutate(expVal = (COUNT / sum(COUNT)) * Item.Rating)
      
      # } else if(contract() == 'Friendly Staff'){ 
      #     df <- mainDF %>% filter(Scale.Type == 'Custodial' &  Item.Name == 'Professional Courtesy' &
      #                                 (DATE <= input$dateRange[2] & DATE >= input$dateRange[1])) %>%
      #         group_by(Terminal.Name,  Item.Rating) %>% summarise(COUNT = n()) %>% mutate(expVal = (COUNT / sum(COUNT)) * Item.Rating)
      
    } else if (contract1() == 'Overall'){
      df <- mainDF %>% filter(Scale.Type == 'Custodial' &
                                (DATE <= input$dateRange1[2] & DATE >= input$dateRange1[1])) %>%
        group_by(Terminal.Name,  Item.Rating) %>% summarise(COUNT = n()) %>% mutate(expVal = (COUNT / sum(COUNT)) * Item.Rating)
    }
    
    df$MEAN <- NA ## creating empty column to insert means and variances
    N <- aggregate(df$COUNT, by = list(df$Terminal.Name), FUN = sum) ### total sample size
    means <- aggregate(df$expVal, by = list(df$Terminal.Name), FUN = sum) ### mean by terminal
    
    # for loop used because I don't know how to insert a constant values into a new column based on the terminal.name
    for(i in 1:length(df$Terminal.Name)){
      if(df$Terminal.Name[i] == 'Terminal A'){
        df$MEAN[i] <- means$x[1]
      } else if(df$Terminal.Name[i] == 'Terminal B'){
        df$MEAN[i] <- means$x[2]
      } else if(df$Terminal.Name[i] == 'Terminal C'){
        df$MEAN[i] <- means$x[3]
      } else if(df$Terminal.Name[i] == 'Terminal D'){
        df$MEAN[i] <- means$x[4]
      } else if(df$Terminal.Name[i] == 'Terminal E'){
        df$MEAN[i] <- means$x[5]
      }
    }
    
    
    df <- df %>% mutate(STD = sqrt((Item.Rating - MEAN)^2 * (COUNT/sum(COUNT)))) ### calculating discrete standard deivations 
    stddev <- aggregate(df$STD, by = list(df$Terminal.Name), FUN = sum) ### std devs by terminal
    
    # combining all summary stats into a dataframe to perform one-way ANOVA analysis
    compare <- data.frame('TERMINAL' = as.factor(unique(df$Terminal.Name)), 'MEAN' = means$x, 'STD.DEV' = stddev$x, TOTAL = N$x)
    anova.table <- with(compare, ind.oneway.second(compare$MEAN, compare$STD.DEV, compare$TOTAL)) ### anova.table output
    
    ## creates an estimate comparison table for each terminal based on the stanardized contrast from the ANOVA table
    ## for a more applicable application, I calculated the estimated percent change between each terminal and used that as output for a numerical representation
    ### could be incorrect because the scores range from 0,1,4 instead of 0,1 which could provide a better representation of comparison
    if(terminal1() == 'A'){
      ab <- (anova.table$standardized.contrasts[1,] / means$x[1])
      ac <- (anova.table$standardized.contrasts[2,] / means$x[1])
      ad <- (anova.table$standardized.contrasts[3,] / means$x[1])
      ae <- (anova.table$standardized.contrasts[4,] / means$x[1])
      
      compare <- rbind(ab[1:3], ac[1:3], ad[1:3], ae[1:3])
      rownames(compare) <- c('A vs. B', "A vs. C", "A vs. D", "A vs. E")
      compare$Terminal <- rownames(compare)
      compare = compare[c(4,1,2,3)]
      names(compare)[2:4] = c('Est. % Diff', 'Lwr Bound', 'Up Bound')
      
    } else if(terminal1() == 'B'){
      ba <- (anova.table$standardized.contrasts[1,] / means$x[1]) * -1 # the opposite of a vs b hence -1
      bc <- (anova.table$standardized.contrasts[5,] / means$x[2])
      bd <- (anova.table$standardized.contrasts[6,] / means$x[2])
      be <- (anova.table$standardized.contrasts[7,] / means$x[2])
      
      compare <- rbind(ba[c(1,3,2)])
      names(compare) <- c('es', 'lower', 'upper')
      compare <- rbind(compare, bc[1:3], bd[1:3], be[1:3])
      
      rownames(compare) <- c('B vs. A', 'B vs. C', 'B vs. D', 'B vs. E')
      compare$Terminal <- rownames(compare)
      compare = compare[c(4,1,2,3)]
      names(compare)[2:4] = c('Est. % Diff', 'Lwr Bound', 'Up Bound')
      
    } else if(terminal1() == 'C'){
      ca <- (anova.table$standardized.contrasts[2,] / means$x[1]) * -1 
      cb <- (anova.table$standardized.contrasts[5,] / means$x[2]) * -1
      cd <- (anova.table$standardized.contrasts[8,] / means$x[3])
      ce <- (anova.table$standardized.contrasts[9,] / means$x[3])
      
      
      compare <- rbind(ca[c(1,3,2)], cb[c(1,3,2)])
      names(compare) <- c('es', 'lower', 'upper')
      compare <- rbind(compare, cd[1:3], ce[1:3])
      
      rownames(compare) <- c('C vs. A', 'C vs. B', 'C vs. D', 'C vs. E')
      compare$Terminal <- rownames(compare)
      compare = compare[c(4,1,2,3)]
      names(compare)[2:4] = c('Est. % Diff', 'Lwr Bound', 'Up Bound')
      
    } else if(terminal1() == 'D'){
      da <- (anova.table$standardized.contrasts[3,] / means$x[1]) * -1
      db <- (anova.table$standardized.contrasts[6,] / means$x[2]) * -1
      dc <- (anova.table$standardized.contrasts[8,] / means$x[3]) * -1
      de <- (anova.table$standardized.contrasts[10,] / means$x[4])
      compare <- rbind(da[c(1,3,2)], db[c(1,3,2)])
      names(compare) <- c('es', 'lower', 'upper')
      compare <- rbind(compare, dc[c(1,3,2)], de[1:3])
      rownames(compare) <- c('D vs. A', 'D vs. B', 'D vs. C', 'D vs. E')
      compare$Terminal <- rownames(compare)
      compare = compare[c(4,1,2,3)]
      names(compare)[2:4] = c('Est. % Diff', 'Lwr Bound', 'Up Bound')
      
    } else if(terminal1() == 'E'){
      ea <- (anova.table$standardized.contrasts[4,] / means$x[1]) * -1
      eb <- (anova.table$standardized.contrasts[7,] / means$x[2]) * -1
      ec <- (anova.table$standardized.contrasts[9,] / means$x[3]) * -1
      ed <- (anova.table$standardized.contrasts[10,] / means$x[4]) * -1
      
      compare <- rbind(ea[c(1,3,2)], eb[c(1,3,2)], ec[c(1,3,2)], ed[c(1,3,2)])
      rownames(compare) <- c('E vs. A', 'E vs. B', 'E vs. C', 'E vs. D')
      compare$Terminal <- rownames(compare)
      compare = compare[c(4,1,2,3)]
      names(compare)[2:4] = c('Est. % Diff', 'Lwr Bound', 'Up Bound')
    }
    
    compare = compare %>% mutate(sig.diff = ifelse(`Lwr Bound` < 0 & `Up Bound` > 0, 'no diff', 'sig diff'))
    
    datatable(compare, options = list(dom = 't',columnDefs = list(list(visible = FALSE, targets = 5)),ordering = FALSE)) %>%
      formatStyle('sig.diff', target = 'row', backgroundColor = styleEqual('no diff', 'grey'),
                  color = styleEqual('no diff', 'white')) %>% formatPercentage(names(compare),2)
  })

## similar logic applied with custodial but maintence is a little bit more nuiance
## friendly staff is only used for the custodial staff. An else if statement towards to end outputs NULL for the app.
output$ComparisonMaint <- renderTable({
  if(contract1() == 'Restroom' | contract1() == 'Gate Lounge'){
    df <- mainDF %>% filter(Scale.Type == 'Maintenance' &  Inspection.Type.Name == contract() &
                              (DATE <= input$dateRange[2] & DATE >= input$dateRange[1])) %>%
      group_by(Terminal.Name,  Item.Rating) %>% summarise(COUNT = n()) %>% mutate(expVal = (COUNT / sum(COUNT)) * Item.Rating)
    
    if (contract1() == 'Overall'){
      df <- mainDF %>% filter(Scale.Type == 'Maintenance' & (Inspection.Type.Name == 'Restroom' | Inspection.Type.Name == 'Gate Lounge') &
                                (DATE <= input$dateRange[2] & DATE >= input$dateRange[1])) %>%
        group_by(Terminal.Name,  Item.Rating) %>% summarise(COUNT = n()) %>% mutate(expVal = (COUNT / sum(COUNT)) * Item.Rating)
    }
    
    df$MEAN <- NA ## creating empty column to insert means and variances
    N <- aggregate(df$COUNT, by = list(df$Terminal.Name), FUN = sum) ### total sample size
    means <- aggregate(df$expVal, by = list(df$Terminal.Name), FUN = sum) ### mean by terminal
    
    for(i in 1:length(df$Terminal.Name)){
      if(df$Terminal.Name[i] == 'Terminal A'){
        df$MEAN[i] <- means$x[1]
      } else if(df$Terminal.Name[i] == 'Terminal B'){
        df$MEAN[i] <- means$x[2]
      } else if(df$Terminal.Name[i] == 'Terminal C'){
        df$MEAN[i] <- means$x[3]
      } else if(df$Terminal.Name[i] == 'Terminal D'){
        df$MEAN[i] <- means$x[4]
      } else if(df$Terminal.Name[i] == 'Terminal E'){
        df$MEAN[i] <- means$x[5]
      }
    }
    df <- df %>% mutate(STD = sqrt((Item.Rating - MEAN)^2 * (COUNT/sum(COUNT)))) ### calculating discrete standard deivations 
    stddev <- aggregate(df$STD, by = list(df$Terminal.Name), FUN = sum) ### std devs by terminal
    
    compare <- data.frame('TERMINAL' = as.factor(unique(df$Terminal.Name)), 'MEAN' = means$x, 'STD.DEV' = stddev$x, TOTAL = N$x)
    anova.table <- with(compare, ind.oneway.second(compare$MEAN, compare$STD.DEV, compare$TOTAL))
    
    # same logic as above
    if(terminal() == 'A'){
      ab <- (anova.table$standardized.contrasts[1,] / means$x[1]) * 100
      ac <- (anova.table$standardized.contrasts[2,] / means$x[1]) * 100
      ad <- (anova.table$standardized.contrasts[3,] / means$x[1]) * 100
      ae <- (anova.table$standardized.contrasts[4,] / means$x[1]) * 100
      
      compare <- rbind(ab[1:3], ac[1:3], ad[1:3], ae[1:3])
      rownames(compare) <- c('A vs. B', "A vs. C", "A vs. D", "A vs. E")
      compare$Terminal <- rownames(compare)
      compare = compare[c(4,1,2,3)]
      compare
      
    } else if(terminal() == 'B'){
      bc <- (anova.table$standardized.contrasts[5,] / means$x[2]) * 100
      bd <- (anova.table$standardized.contrasts[6,] / means$x[2]) * 100
      be <- (anova.table$standardized.contrasts[7,] / means$x[2]) * 100
      ba <- (anova.table$standardized.contrasts[1,] / means$x[1]) * 100 * -1 # the opposite of a vs b hence -1
      
      compare <- rbind(ba[1:3], bc[1:3], bd[1:3], be[1:3])
      rownames(compare) <- c('B vs. A', 'B vs. C', 'B vs. D', 'B vs. E')
      compare$Terminal <- rownames(compare)
      compare = compare[c(4,1,2,3)]
      compare
      
    } else if(terminal() == 'C'){
      ca <- (anova.table$standardized.contrasts[2,] / means$x[1]) * 100 * -1
      cb <- (anova.table$standardized.contrasts[5,] / means$x[2]) * 100 * -1
      cd <- (anova.table$standardized.contrasts[8,] / means$x[3]) * 100
      ce <- (anova.table$standardized.contrasts[9,] / means$x[3]) * 100
      compare <- rbind(ca[1:3], cb[1:3], cd[1:3], ce[1:3])
      rownames(compare) <- c('C vs. A', 'C vs. B', 'C vs. D', 'C vs. E')
      compare$Terminal <- rownames(compare)
      compare = compare[c(4,1,2,3)]
      compare
      
    } else if(terminal() == 'D'){
      da <- (anova.table$standardized.contrasts[3,] / means$x[1]) * 100 * -1
      db <- (anova.table$standardized.contrasts[6,] / means$x[2]) * 100 * -1
      dc <- (anova.table$standardized.contrasts[8,] / means$x[3]) * 100 * -1
      de <- (anova.table$standardized.contrasts[10,] / means$x[4]) * 100
      compare <- rbind(da[1:3], db[1:3], dc[1:3], de[1:3])
      rownames(compare) <- c('D vs. A', 'D vs. B', 'D vs. C', 'D vs. E')
      compare$Terminal <- rownames(compare)
      compare = compare[c(4,1,2,3)]
      compare
      
    } else if(terminal() == 'E'){
      ea <- (anova.table$standardized.contrasts[4,] / means$x[1]) * 100 * -1
      eb <- (anova.table$standardized.contrasts[7,] / means$x[2]) * 100 * -1
      ec <- (anova.table$standardized.contrasts[9,] / means$x[3]) * 100 * -1
      ed <- (anova.table$standardized.contrasts[10,] / means$x[4]) * 100 * -1
      
      compare <- rbind(ea[1:3], eb[1:3], ec[1:3], ed[1:3])
      rownames(compare) <- c('E vs. A', 'E vs. B', 'E vs. C', 'E vs. D')
      compare$Terminal <- rownames(compare)
      compare = compare[c(4,1,2,3)]
      compare
    }
    # } else if(contract() == 'Friendly Staff'){ 
    #     print(NULL)
  }
})

output$ranking <- renderDataTable({  # ranking table that takes into account both custodial and maintence "scores"
  if(contract1() == 'Restroom' | contract1() == 'Gate Lounge'){
    df <- mainDF %>% filter(Inspection.Type.Name == contract1() & DATE <= input$dateRange1[2] & DATE >= input$dateRange1[1]) %>%
      group_by(Terminal.Name,  Item.Rating) %>% summarise(COUNT = n()) %>% mutate(expVal = (COUNT / sum(COUNT)) * Item.Rating)
    # } else if (contract1() == 'Friendly Staff'){
    #     df <- mainDF %>% filter(Item.Name == 'Professional Courtesy' & DATE <= input$dateRange[2] & DATE >= input$dateRange[1]) %>%
    #         group_by(Terminal.Name,  Item.Rating) %>% summarise(COUNT = n()) %>% mutate(expVal = (COUNT / sum(COUNT)) * Item.Rating)
  } else if (contract1() == 'Overall'){
    df <- mainDF %>% filter(DATE <= input$dateRange1[2] & DATE >= input$dateRange1[1] & (Inspection.Type.Name == 'Restroom' | Inspection.Type.Name == 'Gate Lounge')) %>%
      group_by(Terminal.Name,  Item.Rating) %>% summarise(COUNT = n()) %>% mutate(expVal = (COUNT / sum(COUNT)) * Item.Rating)
  }
  
  df$MEAN <- NA ## creating empty column to insert means and variances
  N <- aggregate(df$COUNT, by = list(df$Terminal.Name), FUN = sum) ### total sample size
  means <- aggregate(df$expVal, by = list(df$Terminal.Name), FUN = sum) ### mean by terminal
  
  for(i in 1:length(df$Terminal.Name)){
    if(df$Terminal.Name[i] == 'Terminal A'){
      df$MEAN[i] <- means$x[1]
    } else if(df$Terminal.Name[i] == 'Terminal B'){
      df$MEAN[i] <- means$x[2]
    } else if(df$Terminal.Name[i] == 'Terminal C'){
      df$MEAN[i] <- means$x[3]
    } else if(df$Terminal.Name[i] == 'Terminal D'){
      df$MEAN[i] <- means$x[4]
    } else if(df$Terminal.Name[i] == 'Terminal E'){
      df$MEAN[i] <- means$x[5]
    }
  }
  df <- df %>% mutate(STD = sqrt((Item.Rating - MEAN)^2 * (COUNT/sum(COUNT)))) ### calculating discrete standard deivations 
  stddev <- aggregate(df$STD, by = list(df$Terminal.Name), FUN = sum) ### std devs by terminal
  
  compare <- data.frame('TERMINAL' = as.factor(unique(df$Terminal.Name)), 'MEAN' = means$x, 'STD.DEV' = stddev$x, TOTAL = N$x)
  anova.table <- with(compare, ind.oneway.second(compare$MEAN, compare$STD.DEV, compare$TOTAL))
  
  # same logic as above
  ab <- (anova.table$standardized.contrasts[1,] / means$x[1]) * 100
  ac <- (anova.table$standardized.contrasts[2,] / means$x[1]) * 100
  ad <- (anova.table$standardized.contrasts[3,] / means$x[1]) * 100
  ae <- (anova.table$standardized.contrasts[4,] / means$x[1]) * 100
  
  bc <- (anova.table$standardized.contrasts[5,] / means$x[2]) * 100
  bd <- (anova.table$standardized.contrasts[6,] / means$x[2]) * 100
  be <- (anova.table$standardized.contrasts[7,] / means$x[2]) * 100
  ba <- (anova.table$standardized.contrasts[1,] / means$x[1]) * 100 * -1 # the opposite of a vs b hence -1
  
  ca <- (anova.table$standardized.contrasts[2,] / means$x[1]) * 100 * -1
  cb <- (anova.table$standardized.contrasts[5,] / means$x[2]) * 100 * -1
  cd <- (anova.table$standardized.contrasts[8,] / means$x[3]) * 100
  ce <- (anova.table$standardized.contrasts[9,] / means$x[3]) * 100
  
  da <- (anova.table$standardized.contrasts[3,] / means$x[1]) * 100 * -1
  db <- (anova.table$standardized.contrasts[6,] / means$x[2]) * 100 * -1
  dc <- (anova.table$standardized.contrasts[8,] / means$x[3]) * 100 * -1
  de <- (anova.table$standardized.contrasts[10,] / means$x[4]) * 100
  
  ea <- (anova.table$standardized.contrasts[4,] / means$x[1]) * 100 * -1
  eb <- (anova.table$standardized.contrasts[7,] / means$x[2]) * 100 * -1
  ec <- (anova.table$standardized.contrasts[9,] / means$x[3]) * 100 * -1
  ed <- (anova.table$standardized.contrasts[10,] / means$x[4]) * 100 * -1
  
  
  compare <- round(rbind(
    ab[1:3],ac[1:3],ad[1:3],ae[1:3],bc[1:3],bd[1:3],be[1:3],ba[c(1,3,2)],ca[c(1,3,2)],cb[c(1,3,2)],cd[1:3],ce[1:3],
    da[c(1,3,2)],db[c(1,3,2)],dc[c(1,3,2)],de[1:3],ea[c(1,3,2)],eb[c(1,3,2)],ec[c(1,3,2)],ed[c(1,3,2)]), digits = 2)
  
  compare$Terminal <- c('Terminal A','Terminal A','Terminal A','Terminal A','Terminal B','Terminal B','Terminal B','Terminal B',
                        'Terminal C','Terminal C','Terminal C','Terminal C','Terminal D','Terminal D','Terminal D','Terminal D',
                        'Terminal E','Terminal E','Terminal E','Terminal E')
  compare <- aggregate(compare$es, list(compare$Terminal), FUN=mean) ## for comparison, I took the average percentage estimated change 
  ## the one with the greatest percentage change is considered to be the best terminal -- the largest percentage change is consider to be performing the best 
  compare <- compare[order(-compare$x),] # order by descending
  compare$Rank <- c("1st","2nd","3rd","4th","5th") # seq function called to put "1st, 2nd, 3rd" places
  names(compare)[names(compare) == 'Group.1'] <- 'Terminal' # changing column name to somethign more understanding
  compare = compare[c('Rank', 'Terminal')]
  datatable(compare, options = list(dom = 't',ordering = FALSE), rownames = FALSE)
})

output$sentimentPlot <- renderPlot({
  terminalName = paste("Terminal", as.character(terminal3()))
  df = mainDF
  
  df$category <- ifelse(df$Inspection.Type.Name == 'Gate Lounge', 'Gate Lounge', 
                        ifelse(df$Inspection.Type.Name == 'Restroom', 'Restroom', NA))
  
  texts <- na.omit(df[c('Terminal.Name', 'Item.Comments', 'Scale.Type', 'SCORE', 'DATE', 'category', 'Inspection.Type.Name')]) %>% 
    mutate(index = row_number())
  
  undesired_words <- c("1","2","3","3","4","5","6","7","8","9","0", "#") # removing stray numbers and number sign from analysis
  
  tidy.words = texts %>% filter((Terminal.Name == terminalName) & 
                                  (DATE <= input$dateRange3[2] & DATE >= input$dateRange3[1]) & 
                                  (Inspection.Type.Name == 'Restroom' | Inspection.Type.Name == 'Gate Lounge')) %>%
    group_by(Item.Comments) %>% 
    ungroup() %>% 
    unnest_tokens(word, Item.Comments) %>%
    filter(!word %in% undesired_words) %>%
    anti_join(stop_words)
  
  tidy.words %>%
    inner_join(get_sentiments('bing')) %>%
    group_by(category, sentiment) %>%
    summarize(word_count = n()) %>%
    ungroup() %>%
    mutate(sentiment = reorder(sentiment, word_count)) %>%
    ggplot(aes(sentiment, word_count, fill = category)) +
    geom_col(position = 'dodge') +
    labs(x = NULL, y = 'Word Count') +
    ggtitle('Overall Comment Sentiment by Performance') +
    scale_x_discrete(limits = c('negative', 'positive'))
})

output$sentimentTime <- renderPlot({
  terminalName = paste("Terminal", as.character(terminal3()))
  df = mainDF
  
  df$category <- ifelse(df$Inspection.Type.Name == 'Gate Lounge', 'Gate Lounge', 
                        ifelse(df$Inspection.Type.Name == 'Restroom', 'Restroom', NA))
  
  texts <- na.omit(df[c('Terminal.Name', 'Item.Comments', 'Scale.Type', 'SCORE', 'DATE', 'category', 'Inspection.Type.Name')]) %>% 
    mutate(index = row_number())
  
  undesired_words <- c("1","2","3","3","4","5","6","7","8","9","0", "#") # removing stray numbers and number sign from analysis
  
  tidy.words = texts %>% filter(Terminal.Name == terminalName & (DATE <= input$dateRange3[2] & DATE >= input$dateRange3[1]) &
                                  (Inspection.Type.Name == 'Restroom' | Inspection.Type.Name == 'Gate Lounge')) %>%
    group_by(Item.Comments) %>% 
    ungroup() %>% 
    unnest_tokens(word, Item.Comments) %>%
    filter(!word %in% undesired_words) %>%
    anti_join(stop_words)
  tidy.words %>% 
    inner_join(get_sentiments('nrc')) %>% 
    filter(sentiment %in% c('positive', 'negative')) %>%
    group_by(DATE, sentiment) %>%
    count(DATE, sentiment) %>%
    spread(sentiment, n) %>%
    mutate(polarity = positive - negative,
           ratio = polarity / (positive + negative)) %>%
    mutate(sentiment.class = ifelse(polarity > 0, 'POSITIVE', 'NEGATIVE')) %>%
    ggplot(aes(DATE, polarity, fill = sentiment.class)) +
    geom_bar(stat = 'identity') +
    labs(title = 'Sentiment Over Time',
         xlab = 'Date',
         ylab = 'Polarity',
         caption = 'Polarity = # of positive words - # of negative words',
         fill = 'Sentiment')
})
}

shinyApp(ui = ui, server = server)
