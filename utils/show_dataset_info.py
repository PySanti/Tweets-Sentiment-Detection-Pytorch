def show_dataset_info(X,Y, tag):
    print(f"~~~~~~~~ Dataset : {tag}")
    print(f"Shape del X : {X.shape}")
    print(f"Shape del Y : {Y.shape}")
    print(f"Distribucion de target")
    for tag, val in Y.value_counts().to_dict().items():
        print(f"{tag}   :   {val/len(Y)*100:.2f}")


