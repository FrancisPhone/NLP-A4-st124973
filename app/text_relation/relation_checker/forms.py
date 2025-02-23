from django import forms


class TextRelationForm(forms.Form):
    premise = forms.CharField(widget=forms.Textarea, label="Text A (Premise)")
    hypothesis = forms.CharField(widget=forms.Textarea, label="Text B (Hypothesis)")