#!/bin/bash

if [ ! -d ./.githooks ]; then
    mkdir ./.githooks
    else :
fi
if [ ! -f ./.githooks/pre-commit ]; then
    touch ./.githooks/pre-commit
    else :
fi
#    pre-commitを書きこむ
cat << EOF > ./.githooks/pre-commit
#!/bin/bash

echo "pre-commit!!!!!"

## staed file それぞれに対して
for file in \$(git diff --cached --name-only); do

    ## ipynbの場合
    if [[ \$file == *.ipynb ]]; then
        ## delete output
        jupyter nbconvert --ClearOutputPreprocessor.enabled=True \\
            --ClearOutputPreprocessor.remove_metadata_fields=[]  \\
            --ClearMetadataPreprocessor.enabled=True  \\
            --ClearMetadataPreprocessor.preserve_nb_metadata_mask='[("language_info"),("kernelspec")]' \\
            --to notebook --inplace \${file}

        ## save output
        # jupyter nbconvert \\
            # --ClearMetadataPreprocessor.enabled=True  \\
            # --ClearMetadataPreprocessor.preserve_nb_metadata_mask='[("language_info"),("kernelspec")]' \\
            # --to notebook --inplace \${file}

        ## ipynb -> py script
        # jupyter nbconvert --to script \$file
        # git add \${file%.*}.py


        ## format ipynb
        # black \${file}

        git add .
    fi

    ## ipynbの場合
    if [[ \$file == *.py ]]; then
        ## format py
        black \${file}

        git add .
    fi
done
EOF

#    pre-commitを設定
chmod u+x ./.githooks/*
git config --local core.hooksPath ./.githooks
