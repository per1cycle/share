#include <stdio.h>
#include <libxml/xmlwriter.h>

int main() {
    // Create a new XML writer for memory, with no compression.
    xmlTextWriterPtr writer = xmlNewTextWriterFilename("example.xml", 0);
    if (writer == NULL) {
        fprintf(stderr, "Error creating the xml writer\n");
        return 1;
    }

    // Start the document with the XML version and encoding.
    xmlTextWriterStartDocument(writer, NULL, "UTF-8", NULL);

    // Start the root element.
    xmlTextWriterStartElement(writer, BAD_CAST "catalog");

    // Create a book element.
    xmlTextWriterStartElement(writer, BAD_CAST "book");
    xmlTextWriterWriteAttribute(writer, BAD_CAST "id", BAD_CAST "1");

    // Add title element.
    xmlTextWriterStartElement(writer, BAD_CAST "title");
    xmlTextWriterWriteString(writer, BAD_CAST "The Great Gatsby");
    xmlTextWriterEndElement(writer); // End title

    // Add author element.
    xmlTextWriterStartElement(writer, BAD_CAST "author");
    xmlTextWriterWriteString(writer, BAD_CAST "F. Scott Fitzgerald");
    xmlTextWriterEndElement(writer); // End author

    // Add price element.
    xmlTextWriterStartElement(writer, BAD_CAST "price");
    xmlTextWriterWriteString(writer, BAD_CAST "10.99");
    xmlTextWriterEndElement(writer); // End price

    // End the book element.
    xmlTextWriterEndElement(writer); // End book

    // Create another book element.
    xmlTextWriterStartElement(writer, BAD_CAST "book");
    xmlTextWriterWriteAttribute(writer, BAD_CAST "id", BAD_CAST "2");

    // Add title element.
    xmlTextWriterStartElement(writer, BAD_CAST "title");
    xmlTextWriterWriteString(writer, BAD_CAST "1984");
    xmlTextWriterEndElement(writer); // End title

    // Add author element.
    xmlTextWriterStartElement(writer, BAD_CAST "author");
    xmlTextWriterWriteString(writer, BAD_CAST "George Orwell");
    xmlTextWriterEndElement(writer); // End author

    // Add price element.
    xmlTextWriterStartElement(writer, BAD_CAST "price");
    xmlTextWriterWriteString(writer, BAD_CAST "8.99");
    xmlTextWriterEndElement(writer); // End price

    // End the book element.
    xmlTextWriterEndElement(writer); // End book

    // End the root element.
    xmlTextWriterEndElement(writer); // End catalog

    // End the document.
    xmlTextWriterEndDocument(writer);

    // Free the writer.
    xmlFreeTextWriter(writer);

    printf("XML file created successfully: example.xml\n");
    return 0;
}
