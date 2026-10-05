library(here)
library(tidyverse)
library(irr)
library(patchwork)
options(tibble.print_min = 100)

codebook_dev <- here("toy", "01_codebook_dev", "out")
experiment <- here("toy", "02_experiment", "out")
analysis <- here("toy", "03_analysis")

list_classification_files <- function(dir, folder) {
  list.files(
    file.path(dir, folder),
    pattern = "^classification",
    recursive = TRUE,
    full.names = TRUE
  )
}
read_classification_files <- function(dir, folder) {
  files_list <- list_classification_files(dir, folder)
  map(
    files_list,
    \(path) read_csv(path, show_col_types = FALSE) %>%
      mutate(model = str_extract(basename(path),
                                 "(classifications_)(.*)(\\.csv)",
                                 group=2),
             run = str_extract(path,
                               "(.*)(run_\\d*)(.*)",
                               group=2),
             path = path)
  ) %>% list_rbind()
}

#### Codebookless initial coding ####
folder <- "grimmer_repeated-runs_no-cb"

initial_coding <- read_classification_files(codebook_dev, folder)
initial_coding$model <- str_replace_all(initial_coding$model, "_", " ")
models <- unique(initial_coding$model)
model_order <- models[c(4,5,2,1,3)]

## k's alpha
cat("krippendorf's alpha across 50 runs\n")
walk(model_order, \(mod) {
  df <- initial_coding %>%
    filter(model==mod) %>%
    pivot_wider(
      id_cols="document_id",
      names_from="run",
      values_from="label"
    ) %>%
    select(-document_id)
  alph <- kripp.alpha(t(as.matrix(df)), method = "nominal")$value
  cat(mod, "|", alph, "\n")
})

## agreement by modelxdoc
p1_df <- initial_coding %>%
  summarise(
    p_1 = mean(label == 1),
    .by = c(document_id, model)
  ) %>%
  mutate(agreement = pmax(p_1, 1 - p_1),
         agree_below_80 = agreement <= 0.8) %>%
  mutate(
    at_least_2_below80 = sum(agree_below_80) >= 2,
    n_below_80 = sum(agreement <= .80),
    mean_agreement = mean(agreement),
    .by = document_id)

doc_order <- p1_df %>%
  pivot_wider(id_cols = document_id,
              names_from = model,
              values_from = agreement
  ) %>%
  arrange(
    .data[[model_order[4]]],
    .data[[model_order[1]]],
    .data[[model_order[3]]]
  ) %>%
  pull(document_id)

A <- p1_df %>%
  arrange(model, agreement) %>%
  mutate(model = factor(model, levels = model_order)) %>%
  mutate(rank = row_number(), .by = model) %>%
  ggplot() +
  geom_point(aes(x = rank, y = agreement)) +
  scale_y_continuous(breaks = c(0.6, 0.7, 0.8, 0.9, 1)) +
  facet_wrap(~ model, ncol=1) +
  labs(
    title = "Grimmer",
    x = "Document, ordered within model",
    y = "%agreement\n(minimum 50% for binary outcome)"
  ) +
  theme_bw() +
  theme(panel.grid.minor=element_blank())

B <- p1_df %>%
  mutate(model = factor(model, levels = model_order)) %>%
  mutate(
    document_id = factor(document_id, levels = doc_order),
    doc_position = as.numeric(document_id)) %>%
  ggplot() +
  geom_point(aes(x = doc_position, y = agreement,
                 #color=at_least_2_below80,
                 #color=factor(n_below_80),
                 color=mean_agreement,
                 )) +
  scale_y_continuous(breaks = c(0.6, 0.7, 0.8, 0.9, 1)) +
  scale_color_gradient2(low = "red", high = "green", midpoint = .8) +
  facet_wrap(~ model, ncol=1) +
  labs(
    x = "Document, fixed order",
    y = "% agreement\n(minimum 50% for binary outcome)"
  ) +
  theme_bw() +
  theme(panel.grid.minor=element_blank())

(A + B) +
  plot_layout(guides = "collect", axis_titles = "collect") &
  theme(legend.position = "bottom")

## write out for qual check
walk(models, \(mod) {
  initial_coding %>%
    filter(model == mod) %>%
    mutate(p_1 = mean(label == 1),
           p_0 = mean(label == 0),
           .by = document_id) %>%
    mutate(p_agree = pmax(p_1, 1 - p_1)) %>%
    pivot_wider(
      id_cols = c(document_id, text, p_1, p_agree),
      names_from = run,
      values_from = c(label, explanation),
      names_glue = "{run}_{.value}",
      names_vary = "slowest"
    ) %>%
    write_csv(file.path(codebook_dev, folder,
                        paste0("initial_coding_", str_replace_all(mod, " ", "_"), ".csv")))
})

## Positive and negative cores
agree_ranges <- p1_df %>%
  mutate(
    band = case_when(
      p_1 >= .6 ~ "Yes",
      p_1 <= .4 ~ "No",
      TRUE       ~ "Middle"
    ),
    band = factor(band, levels = c("No", "Middle", "Yes"))
  )

agree_ranges <- p1_df %>%
  mutate(
    band = case_when(
      p_1 >= .9 ~ "Strong yes",
      p_1 >= .8 ~ "Weak yes",
      p_1 >  .2 ~ "Middle",
      p_1 >  .1 ~ "Weak no",
      TRUE      ~ "Strong no"
    ),
    band = factor(
      band,
      levels = c("Strong no", "Weak no", "Middle", "Weak yes", "Strong yes")
    )
  )

mat <- agree_ranges %>%
  select(document_id, model, band) %>%
  pivot_wider(names_from = model, values_from = band) %>%
  filter(
    !if_all(-document_id, ~ . == "Yes"),
    !if_all(-document_id, ~ . == "No"),
    !if_any(-document_id, ~ . == "Middle")
  ) %>%
  mutate(across(-document_id, ~ as.numeric(.))) %>%
  column_to_rownames("document_id") %>%
  as.matrix()

mat2 <- agree_ranges %>%
  select(document_id, model, band) %>%
  pivot_wider(names_from = model, values_from = band) %>%
  filter(
    # !if_all(-document_id, ~ . == "Yes"),
    # !if_all(-document_id, ~ . == "No"),
    !if_any(-document_id, ~ . == "Middle")
  ) %>%
  mutate(across(-document_id, ~ as.numeric(.))) %>%
  column_to_rownames("document_id") %>%
  as.matrix()


pheatmap(
  mat2,
  cluster_cols = FALSE,
  clustering_distance_rows = "manhattan",
  clustering_method = "complete",
  #cutree_rows = 5,
  show_rownames = FALSE
)
#### Initial Coding ####

initial_coding <- read_classification_files(codebook_dev, "grimmer_50/run_1") %>%
  filter(model %in% models) %>%
  pivot_wider(id_cols="document_id",
              names_from="model",
              values_from="label")

combinations <- c(
  list(2:5),
  combn(2:5, 3, simplify = FALSE),
  combn(2:5, 2, simplify = FALSE)
)

combinations <- c(
  list(2:4),
  combn(2:5, 3, simplify = FALSE),
  combn(2:5, 2, simplify = FALSE)
)

results <- map(list(c(2,3,4), c(2,3), c(3,4), c(2,4)), \(x) {
  setNames(
    kripp.alpha(t(as.matrix(initial_coding[, x])), method = "nominal")$value,
    paste(names(initial_coding)[x], collapse = " | ")
  )
})

data.frame(results[[4]])

#### agreement with initial coding ####
no_cb <- read_classification_files(codebook_dev, "lacombe_gunreg_3by") # gemini is streaming, kimi is batched
count(no_cb, model, run)
no_cb %>% distinct(model, run) %>% count(model)
cb <- read_classification_files(experiment, "jung_apply_initial") # batching for gpt
count(cb, model, run)
cb %>% distinct(model, run) %>% count(model)
models <- unique(no_cb$model)

compare_dfs <- map(models, \(mod) {
  
  initial_mod <- initial_coding %>%
    select(document_id, all_of(mod)) %>%
    rename(initial = all_of(mod))
  no_cb_mod <- no_cb %>%
    filter(model == mod) %>%
    select(document_id, run, model, label) %>%
    left_join(initial_mod, by="document_id")
  cb_mod <- cb %>%
    filter(model == mod) %>%
    select(document_id, run, model, label) %>%
    left_join(initial_mod, by="document_id")
  
  ## alpha
  alpha <- list(no_cb = no_cb_mod, cb = cb_mod) %>%
    map(\(df) {
      df %>%
        group_by(model, run) %>%
        summarise(
          alpha = kripp.alpha(
            t(as.matrix(pick(label, initial))),
            method = "nominal"
          )$value,
          .groups = "drop"
        )
    }) %>%
    list_rbind(names_to = "cb")

  ## texts that disagree
  texts <- list(no_cb = no_cb_mod, cb = cb_mod) %>%
    map(\(df) {
      df %>%
        pivot_wider(
          id_cols = c(document_id, initial),
          names_from = run,
          values_from = label
        ) %>%
        mutate(
          n_match = rowSums(across(starts_with("run_"), \(x) x == initial)),
          n_runs = ncol(pick(starts_with("run_"))),
          percent_match = n_match / n_runs,
          model = mod
        )
    }) %>%
    list_rbind(names_to = "cb")
  
  return(list("alpha"=alpha,
              "texts"=texts))
})

compare_dfs %>%
  map_dfr("alpha") %>%
  mutate(run_num = as.numeric(str_extract(run, "\\d+"))) %>%
  ggplot() +
  geom_point(aes(x=model, y=alpha, color=cb),
             position = position_jitter(width = 0.15, height = 0)) +
  scale_y_continuous(breaks=seq(0, 1, 0.1), limits=c(0,1))

order <- compare_dfs[[3]]$texts %>%
  filter(cb == "no_cb") %>%
  arrange(percent_match) %>%
  pull(document_id)
compare_dfs %>%
  map_dfr("texts") %>%
  mutate(document_id = factor(document_id, levels=order)) %>%
  ggplot() +
  geom_point(aes(x=document_id, y=percent_match, color=cb)) +
  facet_grid(model~.)
  
#### 3 by 3 ####
tbt_nocb <- read_classification_files(codebook_dev, "jung_initialcoding_newdata") %>%
  pivot_wider(id_cols=document_id, names_from=model, values_from=label) %>%
  select(-document_id)
kripp.alpha(t(as.matrix(tbt_nocb)), method = "nominal")$value

tbt <- read_classification_files(experiment, "grimmer_3by3") %>%
  mutate(codebook = str_extract(path, "(?<=/codebook_)[^/]+")) %>%
  filter(run == "run_1") %>%
  filter(
    # model != "moonshotai_kimi-k3",
    # codebook !="moonshotai_kimi-k3"
  )
count(tbt, model, codebook, run)
cb %>% distinct(model, run) %>% count(model)

map(unique(tbt$codebook), \(mod) {
  within_model <- tbt %>%
    filter(model == mod) %>%
    pivot_wider(id_cols=document_id, names_from=codebook, values_from=label) %>%
    select(-document_id)
  wm <- kripp.alpha(t(as.matrix(within_model)), method = "nominal")$value
  
  within_cb <- tbt %>%
    filter(codebook == mod) %>%
    pivot_wider(id_cols=document_id, names_from=model, values_from=label) %>%
    select(-document_id)
  wc <- kripp.alpha(t(as.matrix(within_cb)), method = "nominal")$value

  data.frame(model=mod, within_model=wm, within_cb=wc)
}) %>% list_rbind()

