from django.shortcuts import render
from .forms import TextRelationForm
from .utils import check_relation


def index(request):
    result = None

    if request.method == 'POST':
        form = TextRelationForm(request.POST)
        if form.is_valid():
            premise = form.cleaned_data['premise']
            hypothesis = form.cleaned_data['hypothesis']

            relation = check_relation(premise, hypothesis)

            result = {
                'relation': relation
            }
    else:
        form = TextRelationForm()

    return render(request, 'index.html', {'form': form, 'result': result})
