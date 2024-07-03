required_packages = c('tidyr', 'janitor', 'fastDummies', 'forcats', 'stringr')
for (pkg in required_packages){
    suppressPackageStartupMessages(library(pkg, character.only = TRUE, warn.conflicts = FALSE, quietly = TRUE, verbose = FALSE));
}

safe.dir = function(dir_name){
    last_char = substr(dir_name,nchar(dir_name),nchar(dir_name));
    if (last_char != "/") {
        dir_name = paste0(dir_name,"/");
    }
    return(dir_name);
}

print.to.console = function(print_string, ..., flag = TRUE){
    if (flag){
        if (length(list(...))==0){
            for (n in 1:length(print_string)){
                cat(sprintf('%s | %s\n', format(Sys.time(),"%X"), print_string))
            }
        } else {
            n_print_strs = length(print_string)
            if (n_print_strs == 1){
                cat(sprintf('%s | %s\n', format(Sys.time(),"%X"), sprintf(print_string, ...)), sep = "")
            } else {
                for (n in 1:n_print_strs){
                    cat(sprintf('%s | %s\n', format(Sys.time(),"%X"), sprintf(print_string[n], ...)), sep = "")
                }
            }
        }
    }
}

remove.sampling.weights.and.save.to.disk= function(df, sampling_weight_name = NULL, sampling_weight_file = NULL){

    has_sampling_weights = (!is.null(sampling_weight_name)) && ((sampling_weight_name %in% colnames(df)))
    save_to_disk = !is.null(sampling_weight_file)

    if (has_sampling_weights){

        weights_df = df[[sampling_weight_name]];
        stopifnot(all(weights_df>=0));
        as.numeric(weights_df)
        df[[sampling_weight_name]] = NULL;
        print.to.console(sprintf('dropped %s (sampling weights) from data.frame', sampling_weight_name))

        #save sampling weights
        if (save_to_disk){
            write.table(x = weights_df,file = sampling_weight_file, col.names = FALSE, row.names = FALSE, quote = FALSE);
            print.to.console(sprintf('saved sampling weights to %s', sampling_weight_name))
        }
    }
    return(df)
}

get.header.descriptions = function(data, outcome_name, sampling_weight_name="", ordinal_names="", partition_names = ""){

    #make sure there is no sampling weight in helper
    stopifnot(!(sampling_weight_name %in% colnames(data)));

    #create orderings
    orderings = list();
    for (var_name in ordinal_names){
        orderings[[var_name]] = paste0(levels(data[[var_name]]), collapse = "|");
    };

    #create helper variable
    description_df = tibble(header=colnames(data)) %>%
        rowwise() %>%
        mutate(is_outcome = header==outcome_name,
               is_group_attribute = header %in% partition_names,
               numeric_variable = is.numeric(data[[header]]),
               boolean_variable = is.logical(data[[header]]),
               ordinal_variable = header %in% ordinal_names,
               ordering = ifelse(ordinal_variable, orderings[[header]],NA),
               categorical_variable = is.factor(data[[header]]),
               categorical_variable = categorical_variable & (!ordinal_variable)) %>%
        ungroup();

    #basic type checking
    every_variable_has_a_type = description_df %>%
        rowwise() %>%
        mutate(known_type = any(numeric_variable, boolean_variable, categorical_variable, ordinal_variable)) %>%
        select(known_type) %>%
        unlist(use.names=FALSE);

    stopifnot(all(every_variable_has_a_type));

    description_df = description_df %>%
        mutate(type = 1*numeric_variable +
                   2*categorical_variable +
                   3*boolean_variable +
                   4*ordinal_variable);

    description_df$type = factor(description_df$type) %>%
        fct_recode(none = "0",
                   numeric = "1",
                   categorical = "2",
                   boolean = "3",
                   ordinal = "4");

    description_df = description_df %>%
        select(header, is_outcome, is_group_attribute, type, ordering);

    return(description_df);
}

binarize.categorical.variables = function(df,varlist,remove_categorical_variable = TRUE){

    n_rows = nrow(df);

    for (var in varlist) {

        vf = as.matrix(df[,var])

        cats 		= as.array(unique(vf))
        n_cats 	= length(cats)

        newdf  = matrix(NA,nrow=n_rows,ncol=n_cats)

        for (c in 1:n_cats) {
            newdf[,c] = vf==cats[c]
        }

        newdf = as.data.frame(1*newdf);
        names(newdf) = paste0(var,"_eq_",as.character(cats));

        #append binarized variables
        df = cbind(df,newdf);

        #remove binarized variables
        if (remove_categorical_variable) {
            df[[var]] = NULL;
        }
    }

    return(df)
}

move.outcomes.to.end = function(df,outcome_names){

    order = c(setdiff(names(df),outcome_names),outcome_names);
    df = df[,order];
    return(df)
}

move.outcomes.to.start = function(df,outcome_names){

    order = c(outcome_names, setdiff(names(df),outcome_names));
    df = df[,order];
    return(df)
}

binarize.by.thresholds = function(df,var,threshold_vals,threshold_types,remove_original_variable = TRUE){

    n_thresholds 			= length(threshold_vals);
    n_threshold_types = length(threshold_types);

    if ((n_threshold_types==1) && (n_thresholds>1)){
        threshold_types = rep(threshold_types,n_thresholds)
    }

    n_rows 						= nrow(df);
    vf 								= data.matrix(df[[var]])
    tf 								= matrix(NA,nrow=n_rows,ncol=n_thresholds)
    threshold_names  	= rep("",n_thresholds)

    for (t in 1:n_thresholds){

        threshold_type = threshold_types[t]
        if (threshold_vals[t] < 0 ){
            val_name = paste0("n",-threshold_vals[t]);
        } else {
            val_name = paste0(threshold_vals[t]);
        }

        if (threshold_type %in% c("=","eq")){

            tf[,t] 								= vf == threshold_vals[t];
            threshold_names[t] 		= paste0(var,"_eq_",val_name);

        } else if (threshold_type %in% c("<","lt")){

            tf[,t] 								= vf < threshold_vals[t];
            threshold_names[t] 		= paste0(var,"_lt_",val_name);

        } else if (threshold_type %in% c("<=","leq")){

            tf[,t] 								= vf <= threshold_vals[t];
            threshold_names[t] 		= paste0(var,"_leq_",val_name);

        } else if (threshold_type %in% c(">","gt")){

            tf[,t] 								= vf > threshold_vals[t];
            threshold_names[t] 		= paste0(var,"_gt_",val_name);

        } else if (threshold_type %in% c(">=","geq")){

            tf[,t] 								= vf >= threshold_vals[t];
            threshold_names[t] 		= paste0(var,"_geq_",val_name);

        }
    }

    tf = as.data.frame(1*tf);
    names(tf) = threshold_names;

    #append binarized variables
    df = cbind(df,tf);

    #remove binarized variables
    if (remove_original_variable) {
        df[[var]] = NULL;
    }

    return(df)
}

remove.variables.without.variance = function(df){

    vars 		= names(df);
    to_drop = rep(FALSE,length(vars));

    df 	= data.matrix(df);
    P 	= ncol(df);

    for (j in 1:P){
        if (all(df[1,j] == df[,j])){
            to_drop[j] = TRUE;
        }
    }

    cat(sprintf('dropping variables\n%s\n',paste(vars[to_drop],collapse="\n")))

    df 	= as.data.frame(df);
    df 	= df[,!(names(df) %in% vars[to_drop])]

    return(df)

}

convert.logical.to.binary = function(df){
    logical_columns = sapply(df,function(x) is.logical(x));
    df[,logical_columns] = as.data.frame(1*data.matrix(df[,logical_columns]))
    return(df)
}

remove.complements.of.variables = function(df){

    vars 		= names(df);
    to_drop = rep(FALSE,length(vars));

    df 	= data.matrix(df);
    cf  = 1-df;
    P 	= ncol(df);

    for (j in 1:P){

        #if j is a variable that is not dropped
        if (!to_drop[j]) {

            #cat(sprintf('checking for complements of variable %s\n',vars[j]))

            #find all other variables (k) that are not going to be dropped
            other_ind  = setdiff(seq(1,P),j)
            other_ind  = setdiff(other_ind,which(to_drop))

            #mark these variables are to_drop if var_k = 1-var_j
            for (k in other_ind) {
                if (all(df[,k]==cf[,j])) {
                    to_drop[k] = TRUE

                    cat(sprintf('variable %s is a complement of variable %s\n',vars[k],vars[j]))
                }
            }
        }
    }

    cat(sprintf('dropping variables\n%s\n',paste(vars[to_drop],collapse="\n")))

    df 	= as.data.frame(df);
    df 	= df[,!(names(df) %in% vars[to_drop])]

    return(df)
}
