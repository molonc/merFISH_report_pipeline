library(dplyr)
library(ggplot2)
input_dir <-'/Users/hoatran/Documents/BCCRC_projects/merfish/XP2059_FOV25/counting_spots/'
FOV <- '025'
z <- '05'
prefix <- 'merFISH_0'

# Read all csv files histogram and combine them into 1 file for plotting
fns <- list.files(input_dir, pattern = '*_hist.csv')
fns

counts <- tibble::tibble()
for(f in fns){
  df <- data.table::fread(paste0(input_dir,f)) %>% as.data.frame()
  f <- gsub(paste0('_',FOV,'_',z,'_hist.csv'),'',f)
  f <- gsub(prefix,'R',f)
  df$round <- f
  counts <- dplyr::bind_rows(counts, df)
}
dim(counts)
head(counts)

# Counting signals from 250 to 255 intensity values
stat <- counts %>%
  dplyr::group_by(round) %>%
  dplyr::summarise(Count=sum(Count))
stat

# Counting signals=255 intensity values only
stat_255 <- counts %>%
  dplyr::filter(Value==255) %>%
  dplyr::group_by(round) %>%
  dplyr::summarise(Count=sum(Count))
stat_255

p <- ggplot(stat, aes(x=round, y=Count)) +
  geom_bar(stat="identity", fill="steelblue", width=0.4)+
  theme_minimal() +
  labs(title='Counting pixels spots XP2059', x='Round',y='#pixels-spots')
p  

png(paste0(input_dir,"XP2059_spots_count.png"),height = 2*350, width=2*500,res = 2*72)
print(p)
dev.off()

