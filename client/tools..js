function token_cleanup(token) {
    if (currentModel.startsWith("gpt"))
        return token_cleanup_gpt(token);
    else
        return token_cleanup_bert(token);
}

function token_cleanup_bert(token) {
    let clean = token;
    let spaceLeft = true;
    if (token.startsWith('##')) {
        clean = clean.slice(2);
        spaceLeft = false;
    }

    return {
        token,
        clean,
        spaceLeft,
        newLine: false
    };
}

function token_cleanup_gpt(token) {

    let clean = (token.startsWith('Ġ')) ? token.slice(
      1) : ((token.startsWith(
      'Ċ') || token.startsWith('â')) ? " " : token);
    // clean = (token.startsWith('â')) ? '–' : clean;
    // clean = (token.startsWith('ľ')) ? '“' : clean;
    // clean = (token.startsWith('Ŀ')) ? '”' : clean;
    // clean = (token.startsWith('Ļ')) ? "'" : clean;

    try {
        clean = decodeURIComponent(escape(clean));
    } catch {
        console.log(token, '-- token is weird.');
    }

    return {
        token,
        clean,
        spaceLeft: token.startsWith('Ġ'),
        newLine: token.startsWith('Ċ')
    };
}
